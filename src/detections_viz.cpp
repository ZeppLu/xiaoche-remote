#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/VisionInfo.h>

#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


// almost directly copid from jetson-inference/c/detectNet.cpp
cv::Scalar generate_color(uint32_t class_id) {
	// the first color is black, skip that one
	class_id += 1;

	// https://github.com/dusty-nv/pytorch-segmentation/blob/16882772bc767511d892d134918722011d1ea771/datasets/sun_remap.py#L90
#define bitget(byteval, idx)    ((byteval & (1 << idx)) != 0)

	int r = 0;
	int g = 0;
	int b = 0;
	int c = class_id;

	for (int j=0; j<8; j++ ) {
		r = r | (bitget(c, 0) << 7 - j);
		g = g | (bitget(c, 1) << 7 - j);
		b = b | (bitget(c, 2) << 7 - j);
		c = c >> 3;
	}

	return cv::Scalar(b, g, r);
}


void draw_bbox(
		cv::Mat& img,
		const vision_msgs::BoundingBox2D& bbox,
		const std::string& label,
		const cv::Scalar& color,
		int line_thickness, int font_thickness, double font_scale
		) {

	// draw bounding box
	cv::Point upper_left(bbox.center.x - bbox.size_x/2, bbox.center.y - bbox.size_y/2);
	cv::Point lower_right(bbox.center.x + bbox.size_x/2, bbox.center.y + bbox.size_y/2);
	cv::rectangle(img, upper_left, lower_right, color, line_thickness);

	// get the size of label text
	int baseline;
	cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
	// baseline == y-coordinate of the baseline relative to the bottom-most text point
	label_size += cv::Size(0, baseline);

	// draw background of label text
	lower_right = upper_left + cv::Point(label_size);
	cv::rectangle(img, upper_left, lower_right, color, -1);

	// draw class description in white (at upper left corner of bbox)
	cv::Point text_org = upper_left + cv::Point(0, label_size.height-baseline);
	cv::putText(img, label, text_org, cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), font_thickness);
}


void render_and_publish(
		const sensor_msgs::ImageConstPtr& p_image,
		const vision_msgs::Detection2DArrayConstPtr& p_detections,
		const image_transport::Publisher& image_pub,
		const std::vector<std::string>& class_descs,
		int line_thickness, int font_thickness, double font_scale
		) {

	ROS_INFO_THROTTLE(5.0,
			"image delay: %fs; detections delay: %fs",
			(ros::Time::now() - p_image->header.stamp).toSec(),
			(ros::Time::now() - p_detections->header.stamp).toSec()
	);

	// initialize cv_bridge pointer
	cv_bridge::CvImagePtr cv_ptr;
	try {
		cv_ptr = cv_bridge::toCvCopy(*p_image, p_image->encoding);
	} catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
	}

	// actual drawing stuff
	for (const auto& detection : p_detections->detections) {
		const vision_msgs::BoundingBox2D& bbox = detection.bbox;
		// detection.results.size() is guaranteed to be 1 (learnt from detectnet source code)
		const vision_msgs::ObjectHypothesisWithPose& hyp = detection.results[0];

		// use different colors for different classes
		cv::Scalar color = generate_color(hyp.id);

		// construct label of bbox
		std::string desc = class_descs.size() ? class_descs[hyp.id] : "";
		std::stringstream label_stream;
		label_stream << desc << std::setprecision(2) << " (" << hyp.score << ")";

		// draw bounding box and label
		draw_bbox(cv_ptr->image, bbox, label_stream.str(), color, line_thickness, font_thickness, font_scale);
	}

	// publish rendered image
	sensor_msgs::ImagePtr p_rendered = cv_ptr->toImageMsg();
	p_rendered->header.stamp = ros::Time::now();
	image_pub.publish(p_rendered);
}


// callback used to get class descriptions
void fetch_class_descs(const vision_msgs::VisionInfoConstPtr& p_info, std::vector<std::string>& class_descs) {
	const std::string& db = p_info->database_location;
	if (!ros::NodeHandle().getParam(db, class_descs)) {
		ROS_ERROR("failed to obtain class descriptions from %s", db.c_str());
	}
}



int main(int argc, char** argv) {
	ros::init(argc, argv, "detections_viz");

	ros::NodeHandle nh;
	ros::NodeHandle private_nh("~");
	image_transport::ImageTransport it(private_nh);

	std::vector<std::string> class_descs = {};


	// parameters used to help messages matching faster
	// see http://wiki.ros.org/message_filters/ApproximateTime for details
	int detect_delay_lb_ms;
	double age_penalty;
	// TODO: raise exception
	if (!private_nh.getParam("detect_delay_lb_ms", detect_delay_lb_ms)) {
		ROS_ERROR("parameter `detect_delay_lb_ms' not set! exitting...");
		return 0;
	}
	if (!private_nh.getParam("age_penalty", age_penalty)) {
		ROS_ERROR("parameter `age_penalty' not set! exitting...");
		return 0;
	}


	// parameters used in drawing bboxes
	int line_thickness = 1, font_thickness = 1;
	double font_scale = 1.0;
	private_nh.param("line_thickness", line_thickness, line_thickness);
	private_nh.param("font_thickness", font_thickness, font_thickness);
	private_nh.param("font_scale", font_scale, font_scale);


	// try to get vision info, spin once to see if received
	ros::Subscriber vision_info_sub = private_nh.subscribe<vision_msgs::VisionInfo>(
			"vision_info",
			1,
			boost::bind(&fetch_class_descs, _1, boost::ref(class_descs)));


	// rendered images publisher
	image_transport::Publisher image_pub = it.advertise("image_out", 1);


	// subscribers for camera captures & detections output from jetson nano
	// use message_filters to match messages from different topics by timestamps,
	// note that image_transport::SubscriberFilter() is used for sensor_msgs/Image type
	// TODO: what does last argument (queue_size) in these 3 lines do?
	// XXX: when remapping topics, be sure to remap /detections_viz/image_in/compressed
	image_transport::SubscriberFilter image_sub(it, "image_in", 1);
	message_filters::Subscriber<vision_msgs::Detection2DArray> detect_sub(private_nh, "detections", 1);

	// see http://wiki.ros.org/message_filters/ApproximateTime for details
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, vision_msgs::Detection2DArray> SyncPol;
	// queue_size = 10
	message_filters::Synchronizer<SyncPol> sync(SyncPol(10), image_sub, detect_sub);
	// 1s == 1000ms
	sync.setInterMessageLowerBound(ros::Duration(detect_delay_lb_ms / 1000.0));
	sync.setAgePenalty(age_penalty);
	// take const reference of `image_pub' to avoid copying
	sync.registerCallback(
			boost::bind(&render_and_publish, _1, _2, boost::cref(image_pub), boost::cref(class_descs), line_thickness, font_thickness, font_scale));


	ros::spin();

	return 0;
}
