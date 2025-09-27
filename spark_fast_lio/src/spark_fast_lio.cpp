#include "spark_fast_lio.h"

#include <cmath>
#include <stdexcept>

#include <omp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp_components/register_node_macro.hpp>

namespace spark_fast_lio {

SPARKFastLIO2::SPARKFastLIO2(const rclcpp::NodeOptions &options)
    : Node("spark_fast_lio_node", options),
      clock_(get_clock()),
      last_lidar_timestamp_(now()),
      last_imu_timestamp_(now()) {
  preprocessor_  = std::make_shared<Preprocess>();
  imu_processor_ = std::make_shared<ImuProcess>();

  xaxis_point_body_ << LIDAR_SP_LEN, 0.0, 0.0;
  xaxis_point_world_ << LIDAR_SP_LEN, 0.0, 0.0;
  g_base_            = Zero3d;
  mean_acc_stopped_  = Zero3d;
  position_last_     = Zero3d;
  lidar_T_wrt_imu_   = Zero3d;
  lidar_R_wrt_imu_   = Eye3d;
  R_gravity_aligned_ = Eye3d;

  lidar_T_wrt_base_ = Zero3d;
  lidar_R_wrt_base_ = Eye3d;

  path_en_           = declare_parameter<bool>("publish.path_en", true);
  scan_pub_en_       = declare_parameter<bool>("publish.scan_publish_en", false);
  dense_pub_en_      = declare_parameter<bool>("publish.dense_publish_en", false);
  scan_lidar_pub_en_ = declare_parameter<bool>("publish.scan_lidarframe_pub_en", false);
  scan_body_pub_en_  = declare_parameter<bool>("publish.scan_bodyframe_pub_en", false);
  scan_base_pub_en_  = declare_parameter<bool>("publish.scan_baseframe_pub_en", false);

  NUM_MAX_ITERATIONS_ = declare_parameter<int>("max_iteration", 4);

  map_file_path_ = declare_parameter<std::string>("map_file_path", "");
  save_dir_      = declare_parameter<std::string>("common.save_dir", "");
  sequence_name_ = declare_parameter<std::string>("common.sequence_name", "");
  map_frame_     = declare_parameter<std::string>("common.map_frame", "odom");
  lidar_frame_   = declare_parameter<std::string>("common.lidar_frame", "lidar");
  base_frame_    = declare_parameter<std::string>("common.base_frame", "");
  imu_frame_     = declare_parameter<std::string>("common.imu_frame", "imu");
  viz_frame_     = declare_parameter<std::string>("common.visualization_frame", "imu");
  time_sync_en_  = declare_parameter<bool>("common.time_sync_en", false);

  filter_size_map_min_ = declare_parameter<double>("filter_size_map", 0.5);
  cube_len_            = declare_parameter<double>("cube_side_length", 200.0);
  det_range_           = declare_parameter<double>("mapping.det_range", 300.0);
  fov_deg_             = declare_parameter<double>("mapping.fov_degree", 360.0);
  gyr_cov_             = declare_parameter<double>("mapping.gyr_cov", 0.1);
  acc_cov_             = declare_parameter<double>("mapping.acc_cov", 0.1);
  b_gyr_cov_           = declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
  b_acc_cov_           = declare_parameter<double>("mapping.b_acc_cov", 0.0001);

  enable_gravity_alignment_ =
      declare_parameter<bool>("gravity_alignment.enable_gravity_alignment", true);
  acc_diff_thr_          = declare_parameter<double>("gravity_alignment.acc_diff_thr", 0.2);
  num_moving_frames_thr_ = declare_parameter<int>("gravity_alignment.num_moving_frames_thr", 20);
  num_gravity_measurements_thr_ =
      declare_parameter<int>("gravity_alignment.num_gravity_measurements_thr", 20);

  verbose_     = declare_parameter<bool>("verbose", false);
  pcl_verbose_ = declare_parameter<bool>("pcl_verbose", true);
  if (!pcl_verbose_) {
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  }

  runtime_pos_log_      = declare_parameter<bool>("runtime_pos_log_enable", false);
  extrinsic_est_en_     = declare_parameter<bool>("mapping.extrinsic_est_en", false);
  extrinsics_timeout_s_ = declare_parameter<double>("extrinsics_timeout_s", 10.0);
  pcd_save_en_          = declare_parameter<bool>("pcd_save.pcd_save_en", false);
  pcd_save_interval_    = declare_parameter<int>("pcd_save.interval", -1);

  point_filter_num_ = declare_parameter<int>("point_filter_num", 4);

  // extrinT_ and extrinR_ are sized 3 and 9 respectively
  extrinT_ = declare_parameter<std::vector<double>>("mapping.extrinsic_T", extrinT_);
  extrinR_ = declare_parameter<std::vector<double>>("mapping.extrinsic_R", extrinR_);

  auto g_vec = declare_parameter<std::vector<double>>("gravity_alignment.g_base", {0.0, 0.0, -1.0});
  g_base_ << g_vec[0], g_vec[1], g_vec[2];

  auto sensor_qos = rclcpp::SensorDataQoS();
  sub_lidar_      = create_subscription<sensor_msgs::msg::PointCloud2>(
      "lidar",
      sensor_qos,
      std::bind(&SPARKFastLIO2::standardLiDARCallback, this, std::placeholders::_1));

#if defined(LIVOX_ROS_DRIVER_FOUND) && LIVOX_ROS_DRIVER_FOUND
  sub_lidar_livox_ = create_subscription<livox_ros_driver2::msg::CustomMsg>(
      "lidar",
      sensor_qos,
      std::bind(&SPARKFastLIO2::livoxLidarCallback, this, std::placeholders::_1));
#endif

  sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      "imu", sensor_qos, std::bind(&SPARKFastLIO2::imuCallback, this, std::placeholders::_1));

  rclcpp::QoS qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
  pub_cloud_full_  = create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered", qos);
  pub_cloud_lidar_ = create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered_lidar", qos);
  pub_cloud_body_  = create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered_body", qos);
  pub_cloud_base_  = create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered_base", qos);

  pub_odom_                 = create_publisher<nav_msgs::msg::Odometry>("odometry", qos);
  pub_path_                 = create_publisher<nav_msgs::msg::Path>("path", qos);
  path_msg_.header.frame_id = map_frame_;

  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
  tf_buffer_      = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_    = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  preprocessor_        = std::make_shared<Preprocess>();
  preprocessor_->blind = declare_parameter<double>("preprocess.blind", 0.01);
  preprocessor_->blind_for_human_pilots =
      declare_parameter<double>("preprocess.blind_for_human_pilots", 1.5);
  preprocessor_->lidar_type =
      declare_parameter<int>("preprocess.lidar_type", static_cast<int>(AVIA));
  preprocessor_->N_SCANS = declare_parameter<int>("preprocess.scan_line", 16);
  preprocessor_->time_unit =
      declare_parameter<int>("preprocess.timestamp_unit", static_cast<int>(US));
  preprocessor_->SCAN_RATE        = declare_parameter<int>("preprocess.scan_rate", 10);
  preprocessor_->point_filter_num = declare_parameter<int>("point_filter_num_for_preprocessing", 1);

  imu_processor_ = std::make_shared<ImuProcess>();
  if (extrinT_.size() == 3 && extrinR_.size() == 9) {
    Eigen::Vector3d t_lidar(extrinT_[0], extrinT_[1], extrinT_[2]);
    Eigen::Matrix3d R_lidar;
    R_lidar << extrinR_[0], extrinR_[1], extrinR_[2], extrinR_[3], extrinR_[4], extrinR_[5],
        extrinR_[6], extrinR_[7], extrinR_[8];
    imu_processor_->set_extrinsic(t_lidar, R_lidar);
  }

  imu_processor_->set_gyr_cov(Eigen::Vector3d(gyr_cov_, gyr_cov_, gyr_cov_));
  imu_processor_->set_acc_cov(Eigen::Vector3d(acc_cov_, acc_cov_, acc_cov_));
  imu_processor_->set_gyr_bias_cov(Eigen::Vector3d(b_gyr_cov_, b_gyr_cov_, b_gyr_cov_));
  imu_processor_->set_acc_bias_cov(Eigen::Vector3d(b_acc_cov_, b_acc_cov_, b_acc_cov_));

  down_size_filter_.setLeafSize(filter_size_map_min_, filter_size_map_min_, filter_size_map_min_);

  double epsi[23];
  for (int i = 0; i < 23; ++i) epsi[i] = 0.001;

  kf_.init_dyn_share(
      get_f,
      df_dx,
      df_dw,
      // we use a lambda so we can call a member function
      std::bind(&SPARKFastLIO2::calcHModel, this, std::placeholders::_1, std::placeholders::_2),
      NUM_MAX_ITERATIONS_,
      epsi);

  cloud_undistort_.reset(new PointCloudXYZI());
  feats_undistort_.reset(new PointCloudXYZI());
  feats_down_body_.reset(new PointCloudXYZI());
  feats_down_world_.reset(new PointCloudXYZI());
  surface_normals_.reset(new PointCloudXYZI(100000, 1));
  normvec_.reset(new PointCloudXYZI(100000, 1));
  laser_cloud_ori_.reset(new PointCloudXYZI(100000, 1));
  corr_normvec_.reset(new PointCloudXYZI(100000, 1));
  cloud_to_be_saved_.reset(new PointCloudXYZI());

  if (!base_frame_.empty()) {
    if (!lookupBaseExtrinsics(lidar_T_wrt_base_, lidar_R_wrt_base_)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to lookup transform.");
      return;
    }
  }

  main_loop_timer_ =
      create_wall_timer(std::chrono::milliseconds(1), std::bind(&SPARKFastLIO2::main, this));

  if ((preprocessor_->point_filter_num != 1 && point_filter_num_ > 1)) {
    RCLCPP_WARN(this->get_logger(),
                "Points may be too sparse. Set 'preprocessor_->point_filter_num = 1' and tune "
                "'point_filter_num_' instead.");
  }

  RCLCPP_INFO(this->get_logger(), "SPARKFastLIO2 constructed");
}

// Outputs rotation matrix that aligns a to b, i.e., R such that R * g_a = g_b
M3D SPARKFastLIO2::computeRelativeRotation(const Eigen::Vector3d &g_a, const Eigen::Vector3d &g_b) {
  Eigen::Vector3d g_a_norm = g_a.normalized();
  Eigen::Vector3d g_b_norm = g_b.normalized();

  Eigen::Vector3d axis = g_a_norm.cross(g_b_norm);
  double cos_theta     = g_a_norm.dot(g_b_norm);

  if (std::fabs(1.0 - cos_theta) < 1e-3) {
    return Eigen::Matrix3d::Identity();
  }

  // Degenerate condition a = -b
  // Compute cross product with any arbitrary nonparallel vector,
  // i.e., Eigen::Vector3d(1, 2, 3)
  if (std::fabs(1.0 + cos_theta) < 1e-3) {
    Eigen::Vector3d perturbed = g_a_norm + Eigen::Vector3d(1, 2, 3);
    axis                      = g_a_norm.cross(perturbed);

    if (axis.norm() < 1e-6) {
      perturbed = g_a_norm + Eigen::Vector3d(3, 2, 1);
      axis      = g_a_norm.cross(perturbed);
    }

    axis.normalize();
    return Eigen::AngleAxisd(M_PI, axis).toRotationMatrix();
  } else {
    axis.normalize();
    double theta = std::acos(cos_theta);

    Eigen::Quaterniond q(Eigen::AngleAxisd(theta, axis));

    return q.toRotationMatrix();
  }
}

bool SPARKFastLIO2::lookupBaseExtrinsics(V3D &lidar_T_wrt_base, M3D &lidar_R_wrt_base) {
  RCLCPP_INFO(this->get_logger(),
              "Looking up transform from %s -> %s",
              base_frame_.c_str(),
              lidar_frame_.c_str());

  const auto lookup_time = rclcpp::Time(0);
  bool has_transform     = false;
  std::string err_str;
  auto start_time          = this->now();
  rclcpp::Duration timeout = rclcpp::Duration::from_seconds(extrinsics_timeout_s_);
  rclcpp::Rate rate(10.0);  // Just 10 Hz works

  while (rclcpp::ok()) {
    if (tf_buffer_->canTransform(
            base_frame_, lidar_frame_, lookup_time, tf2::durationFromSec(0.0), &err_str)) {
      RCLCPP_INFO_STREAM(this->get_logger(), "\033[1;32mExtrinsics detected.\033[1;0m");
      has_transform = true;
      break;
    }

    const auto time_since_start = now() - start_time;
    if (extrinsics_timeout_s_ > 0.0 && time_since_start > timeout) {
      RCLCPP_ERROR_STREAM(this->get_logger(),
                          "Timeout after "
                              << timeout.seconds() << " seconds waiting for transform from '"
                              << lidar_frame_ << "' to '" << base_frame_ << "': " << err_str);
      break;
    }

    RCLCPP_WARN_STREAM_SKIPFIRST_THROTTLE(get_logger(),
                                          *clock_,
                                          5000,
                                          "Waiting for transform from '" << lidar_frame_ << "' to '"
                                                                         << base_frame_
                                                                         << "': " << err_str);

    rate.sleep();
  }

  if (!has_transform) {
    return has_transform;
  }

  const auto &transform = tf_buffer_->lookupTransform(base_frame_, lidar_frame_, lookup_time);
  lidar_T_wrt_base(0)   = transform.transform.translation.x;
  lidar_T_wrt_base(1)   = transform.transform.translation.y;
  lidar_T_wrt_base(2)   = transform.transform.translation.z;

  Eigen::Quaterniond q(transform.transform.rotation.w,
                       transform.transform.rotation.x,
                       transform.transform.rotation.y,
                       transform.transform.rotation.z);

  lidar_R_wrt_base = q.toRotationMatrix();

  RCLCPP_INFO(this->get_logger(),
              "Translation: [%.3f, %.3f, %.3f]",
              lidar_T_wrt_base(0),
              lidar_T_wrt_base(1),
              lidar_T_wrt_base(2));

  RCLCPP_INFO(this->get_logger(),
              "Rotation (Quaternion): [%.3f, %.3f, %.3f, %.3f]",
              q.x(),
              q.y(),
              q.z(),
              q.w());

  return has_transform;
}

void SPARKFastLIO2::pointBodyToWorld(PointType const *const pi,
                                     PointType *const po,
                                     const state_ikfom &s) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

  po->x         = p_global(0);
  po->y         = p_global(1);
  po->z         = p_global(2);
  po->intensity = pi->intensity;
}

void SPARKFastLIO2::pclPointBodyToWorld(PointType const *const pi, PointType *const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(latest_state_.rot *
                   (latest_state_.offset_R_L_I * p_body + latest_state_.offset_T_L_I) +
               latest_state_.pos);

  po->x         = p_global(0);
  po->y         = p_global(1);
  po->z         = p_global(2);
  po->intensity = pi->intensity;
}

void SPARKFastLIO2::pclPointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(latest_state_.offset_R_L_I * p_body_lidar + latest_state_.offset_T_L_I);

  po->x         = p_body_imu(0);
  po->y         = p_body_imu(1);
  po->z         = p_body_imu(2);
  po->intensity = pi->intensity;
}

void SPARKFastLIO2::pclPointBodyLidarToBase(PointType const *const pi, PointType *const po) {
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_base(lidar_R_wrt_base_ * p_body_lidar + lidar_T_wrt_base_);

  po->x         = p_body_base(0);
  po->y         = p_body_base(1);
  po->z         = p_body_base(2);
  po->intensity = pi->intensity;
}

void SPARKFastLIO2::pclPointIMUToLiDAR(PointType const *const pi, PointType *const po) {
  V3D p_body_imu(pi->x, pi->y, pi->z);
  V3D p_body_lidar(latest_state_.offset_R_L_I.inverse() *
                   (p_body_imu - latest_state_.offset_T_L_I));

  po->x         = p_body_lidar(0);
  po->y         = p_body_lidar(1);
  po->z         = p_body_lidar(2);
  po->intensity = pi->intensity;
}

void SPARKFastLIO2::pclPointIMUToBase(PointType const *const pi, PointType *const po) {
  static const auto &offset_R_B_I = latest_state_.offset_R_L_I * lidar_R_wrt_base_.inverse();
  static const auto &offset_T_B_I =
      -1 * offset_R_B_I * lidar_T_wrt_base_ + latest_state_.offset_T_L_I;

  V3D p_body_imu(pi->x, pi->y, pi->z);
  V3D p_body_base(offset_R_B_I.inverse() * (p_body_imu - offset_T_B_I));

  po->x         = p_body_base(0);
  po->y         = p_body_base(1);
  po->z         = p_body_base(2);
  po->intensity = pi->intensity;
}

void SPARKFastLIO2::collectRemovedPoints() {
  PointVector points_history;
  ikd_tree_.acquire_removed_points(points_history);
}

void SPARKFastLIO2::standardLiDARCallback(const sensor_msgs::msg::PointCloud2 &msg) {
  std::lock_guard<std::mutex> lk(buffer_mutex_);
  scan_count_++;
  rclcpp::Time msg_time = msg.header.stamp;

  if (msg_time < last_lidar_timestamp_) {
    RCLCPP_ERROR(get_logger(), "Lidar loopback detected, clearing buffers");
    lidar_buffer_.clear();
  }
  last_lidar_timestamp_ = msg_time;

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  preprocessor_->process(msg, ptr);

  lidar_buffer_.push_back(ptr);
  time_buffer_.push_back(msg_time.seconds());

  sig_buffer_.notify_all();
}

#if defined(LIVOXROS_DRIVER_FOUND) && LIVOX_ROS_DRIVER_FOUND
void SPARKFastLIO2::livoxLiDARCallback(
    const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr msg) {
  static bool timediff_set_flg = false;

  std::lock_guard<std::mutex> lk(buffer_mutex_);
  scan_count_++;
  rclcpp::Time msg_time = msg.header.stamp;

  if (msg_time < last_lidar_timestamp_) {
    RCLCPP_ERROR(get_logger(), "Livox loopback, clearing buffers");
    lidar_buffer_.clear();
  }
  last_lidar_timestamp_ = msg_time;

  const auto diff_s = std::abs((last_imu_timestamp_ - last_lidar_timestamp_).seconds());
  if (!time_sync_en_ && diff_s > 10.0 && !imu_buffer_.empty() && !lidar_buffer_.empty()) {
    RCLCPP_WARN_STREAM(this->get_logger(),
                       "IMU and LiDAR not Synced, IMU time: "
                           << last_imu_timestamp_.nanoseconds()
                           << ", lidar header time: " << last_lidar_timestamp_.nanoseconds());
  }

  if (time_sync_en_ && !timediff_set_flg && diff_s > 1.0 && !imu_buffer.empty()) {
    timediff_set_flg        = true;
    timediff_lidar_wrt_imu_ = last_lidar_timestamp_.nanseconds() + static_cast<int64_t>(1.0e8) -
                              last_imu_timestamp_.nanoseconds();
    RCLCPP_INFO_STREAM(
        this->get_logger(),
        "Self sync IMU and LiDAR, time diff is " << timediff_lidar_wrt_imu_ << "[ns]");
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  preprocessor_->process(msg, ptr);

  lidar_buffer_.push_back(ptr);
  time_buffer_.push_back(msg_time.seconds());

  sig_buffer_.notify_all();
}
#endif

void SPARKFastLIO2::imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg) {
  ++publish_count_;

  rclcpp::Time stamp = msg->header.stamp;
  std::lock_guard<std::mutex> lk(buffer_mutex_);

  auto imu_input = std::make_shared<sensor_msgs::msg::Imu>(*msg);
  if (time_sync_en_ && std::abs(timediff_lidar_wrt_imu_) > static_cast<int64_t>(1.0e8)) {
    stamp += rclcpp::Duration::from_nanoseconds(timediff_lidar_wrt_imu_);
    imu_input->header.stamp = stamp;
  }

  if (stamp < last_imu_timestamp_) {
    RCLCPP_WARN_STREAM(get_logger(),
                       "IMU loopback, clearing buffers (previous: "
                           << last_imu_timestamp_.nanoseconds()
                           << " vs. received: " << stamp.nanoseconds() << " [ns]");
    imu_buffer_.clear();
    kf_for_preintegration_.reset();
  }
  last_imu_timestamp_ = stamp;

  if (kf_for_preintegration_.has_value()) {
    integrateIMU(*kf_for_preintegration_, *imu_input);
  }

  imu_buffer_.push_back(imu_input);
  sig_buffer_.notify_all();
}

void SPARKFastLIO2::integrateIMU(esekfom::esekf<state_ikfom, 12, input_ikfom> &state,
                                 const sensor_msgs::msg::Imu &msg) {
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  static std::deque<sensor_msgs::msg::Imu> imu_queue;
  imu_queue.push_back(msg);

  if (imu_queue.size() < 2) {
    return;
  }

  // Assume that timestamps are sufficiently close and ascending order
  double dt = rclcpp::Time(imu_queue[1].header.stamp).seconds() -
              rclcpp::Time(imu_queue[0].header.stamp).seconds();

  if (dt <= 0) {
    RCLCPP_ERROR(this->get_logger(), "IMU timestamps must be in ascending order!");
    imu_queue.pop_front();
    return;
  }

  auto integrated_state = imu_processor_->IntegrateIMU(imu_queue, state);
  const auto &stamp     = imu_queue[1].header.stamp;
  imu_queue.pop_front();

  integrated_state.pos = R_gravity_aligned_ * integrated_state.pos;
  integrated_state.rot = R_gravity_aligned_ * integrated_state.rot;

  publishOdometry(integrated_state, stamp);
}

void SPARKFastLIO2::calcHModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
  double match_start = omp_get_wtime();
  laser_cloud_ori_->clear();
  corr_normvec_->clear();
  total_residual_ = 0.0;

  /** closest surface search and residual computation **/
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < feats_down_size_; i++) {
    PointType &point_body  = feats_down_body_->points[i];
    PointType &point_world = feats_down_world_->points[i];

    /* transform to world frame */
    V3D p_body(point_body.x, point_body.y, point_body.z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
    point_world.x         = p_global(0);
    point_world.y         = p_global(1);
    point_world.z         = p_global(2);
    point_world.intensity = point_body.intensity;

    vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

    auto &points_near = nearest_points_[i];

    if (ekfom_data.converge) {
      /** Find the closest surfaces in the map **/
      ikd_tree_.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
      point_selected_surf_[i] = points_near.size() < NUM_MATCH_POINTS        ? false
                                : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                             : true;
    }

    if (!point_selected_surf_[i]) continue;

    VF(4) pabcd;
    point_selected_surf_[i] = false;
    if (esti_plane(pabcd, points_near, 0.1f)) {
      float pd2 =
          pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
      float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

      if (s > 0.9) {
        point_selected_surf_[i]       = true;
        normvec_->points[i].x         = pabcd(0);
        normvec_->points[i].y         = pabcd(1);
        normvec_->points[i].z         = pabcd(2);
        normvec_->points[i].intensity = pd2;
        res_last_[i]                  = abs(pd2);
      }
    }
  }

  effect_feat_num_ = 0;

  for (int i = 0; i < feats_down_size_; i++) {
    if (point_selected_surf_[i]) {
      laser_cloud_ori_->points[effect_feat_num_] = feats_down_body_->points[i];
      corr_normvec_->points[effect_feat_num_]    = normvec_->points[i];
      total_residual_ += res_last_[i];
      effect_feat_num_++;
    }
  }

  if (effect_feat_num_ < 1) {
    ekfom_data.valid = false;
    RCLCPP_WARN(this->get_logger(), "No Effective Points!");
    return;
  }

  res_mean_last_ = total_residual_ / effect_feat_num_;
  match_time_ += omp_get_wtime() - match_start;
  double solve_start = omp_get_wtime();

  /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
  ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
  ekfom_data.h.resize(effect_feat_num_);

  for (int i = 0; i < effect_feat_num_; i++) {
    const PointType &laser_p = laser_cloud_ori_->points[i];
    V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
    M3D point_be_crossmat;
    point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
    V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);

    /*** get the normal vector of closest surface/corner ***/
    const PointType &norm_p = corr_normvec_->points[i];
    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

    /*** calculate the Measuremnt Jacobian matrix H ***/
    V3D C(s.rot.conjugate() * norm_vec);
    V3D A(point_crossmat * C);
    if (extrinsic_est_en_) {
      V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // s.rot.conjugate()*norm_vec);
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A),
          VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
    } else {
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0;
    }

    /*** Measuremnt: distance to the closest surface/corner ***/
    ekfom_data.h(i) = -norm_p.intensity;
  }
  solve_time_ += omp_get_wtime() - solve_start;
}

void SPARKFastLIO2::lasermapFovSegment() {
  static bool localmap_initialized = false;

  cub_needrm_.clear();
  kdtree_delete_counter_ = 0;
  kdtree_delete_time_    = 0.0;
  pointBodyToWorld(xaxis_point_body_, xaxis_point_world_, latest_state_);

  // `lidar_xyz`: LiDAR position w.r.t. world frame
  V3D lidar_xyz = kf_.get_lidar_position();
  if (!localmap_initialized) {
    for (int i = 0; i < 3; i++) {
      localmap_points_.vertex_min[i] = lidar_xyz(i) - cube_len_ / 2.0;
      localmap_points_.vertex_max[i] = lidar_xyz(i) + cube_len_ / 2.0;
    }
    localmap_initialized = true;
    return;
  }

  float dist_to_map_edge[3][2];
  bool need_move = false;
  for (int i = 0; i < 3; i++) {
    dist_to_map_edge[i][0] = fabs(lidar_xyz(i) - localmap_points_.vertex_min[i]);
    dist_to_map_edge[i][1] = fabs(lidar_xyz(i) - localmap_points_.vertex_max[i]);
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * det_range_ ||
        dist_to_map_edge[i][1] <= MOV_THRESHOLD * det_range_)
      need_move = true;
  }
  if (!need_move) return;
  BoxPointType new_localmap_points, tmp_boxpoints;
  new_localmap_points = localmap_points_;
  float mov_dist      = max((cube_len_ - 2.0 * MOV_THRESHOLD * det_range_) * 0.5 * 0.9,
                       static_cast<double>(det_range_ * (MOV_THRESHOLD - 1)));
  for (int i = 0; i < 3; i++) {
    tmp_boxpoints = localmap_points_;
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * det_range_) {
      new_localmap_points.vertex_max[i] -= mov_dist;
      new_localmap_points.vertex_min[i] -= mov_dist;
      tmp_boxpoints.vertex_min[i] = localmap_points_.vertex_max[i] - mov_dist;
      cub_needrm_.push_back(tmp_boxpoints);
    } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * det_range_) {
      new_localmap_points.vertex_max[i] += mov_dist;
      new_localmap_points.vertex_min[i] += mov_dist;
      tmp_boxpoints.vertex_max[i] = localmap_points_.vertex_min[i] + mov_dist;
      cub_needrm_.push_back(tmp_boxpoints);
    }
  }
  localmap_points_ = new_localmap_points;

  collectRemovedPoints();
  double delete_begin = omp_get_wtime();
  if (cub_needrm_.size() > 0) {
    kdtree_delete_counter_ = ikd_tree_.Delete_Point_Boxes(cub_needrm_);
  }
  kdtree_delete_time_ = omp_get_wtime() - delete_begin;
}

void SPARKFastLIO2::mapIncremental() {
  PointVector PointToAdd;
  PointVector PointNoNeedDownsample;
  PointToAdd.reserve(feats_down_size_);
  PointNoNeedDownsample.reserve(feats_down_size_);

  for (int i = 0; i < feats_down_size_; i++) {
    // transform to world frame
    pointBodyToWorld(
        &(feats_down_body_->points[i]), &(feats_down_world_->points[i]), latest_state_);

    // decide if we need to add to map
    if (!nearest_points_[i].empty() && flg_EKF_inited_) {
      const PointVector &points_near = nearest_points_[i];
      bool need_add                  = true;

      PointType mid_point;
      mid_point.x =
          std::floor(feats_down_world_->points[i].x / filter_size_map_min_) * filter_size_map_min_ +
          0.5 * filter_size_map_min_;
      mid_point.y =
          std::floor(feats_down_world_->points[i].y / filter_size_map_min_) * filter_size_map_min_ +
          0.5 * filter_size_map_min_;
      mid_point.z =
          std::floor(feats_down_world_->points[i].z / filter_size_map_min_) * filter_size_map_min_ +
          0.5 * filter_size_map_min_;

      float dist = calc_dist(feats_down_world_->points[i], mid_point);
      if (std::fabs(points_near[0].x - mid_point.x) > 0.5f * filter_size_map_min_ &&
          std::fabs(points_near[0].y - mid_point.y) > 0.5f * filter_size_map_min_ &&
          std::fabs(points_near[0].z - mid_point.z) > 0.5f * filter_size_map_min_) {
        PointNoNeedDownsample.push_back(feats_down_world_->points[i]);
        continue;
      }

      for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
        if (points_near.size() < NUM_MATCH_POINTS) break;
        if (calc_dist(points_near[readd_i], mid_point) < dist) {
          need_add = false;
          break;
        }
      }
      if (need_add) {
        PointToAdd.push_back(feats_down_world_->points[i]);
      }
    } else {
      // no nearest points, or not in EKF inited
      PointToAdd.push_back(feats_down_world_->points[i]);
    }
  }

  double st_time  = omp_get_wtime();
  add_point_size_ = ikd_tree_.Add_Points(PointToAdd, true);
  ikd_tree_.Add_Points(PointNoNeedDownsample, false);

  add_point_size_          = PointToAdd.size() + PointNoNeedDownsample.size();
  kdtree_incremental_time_ = omp_get_wtime() - st_time;
}

void SPARKFastLIO2::publishOdometry(const state_ikfom &state, const rclcpp::Time &stamp) {
  odomAftMapped_.header.frame_id = map_frame_;
  odomAftMapped_.header.stamp    = stamp;

  setPoseStamp(state, odomAftMapped_.pose, viz_frame_);  // our template function

  if (viz_frame_ == "lidar") {
    odomAftMapped_.child_frame_id = lidar_frame_;
  } else if (viz_frame_ == "base") {
    odomAftMapped_.child_frame_id = base_frame_;
  } else if (viz_frame_ == "imu") {
    odomAftMapped_.child_frame_id = imu_frame_;
  } else {
    throw std::invalid_argument("Invalid visualization frame has been given");
  }

  // fill the covariance
  auto P = kf_.get_P();
  for (int i = 0; i < 6; i++) {
    int k                                     = (i < 3) ? (i + 3) : (i - 3);
    odomAftMapped_.pose.covariance[i * 6 + 0] = P(k, 3);
    odomAftMapped_.pose.covariance[i * 6 + 1] = P(k, 4);
    odomAftMapped_.pose.covariance[i * 6 + 2] = P(k, 5);
    odomAftMapped_.pose.covariance[i * 6 + 3] = P(k, 0);
    odomAftMapped_.pose.covariance[i * 6 + 4] = P(k, 1);
    odomAftMapped_.pose.covariance[i * 6 + 5] = P(k, 2);
  }

  // publish
  pub_odom_->publish(odomAftMapped_);

  geometry_msgs::msg::TransformStamped transform_stamped;
  transform_stamped.header.stamp    = odomAftMapped_.header.stamp;
  transform_stamped.header.frame_id = map_frame_;
  transform_stamped.child_frame_id  = odomAftMapped_.child_frame_id;

  transform_stamped.transform.translation.x = odomAftMapped_.pose.pose.position.x;
  transform_stamped.transform.translation.y = odomAftMapped_.pose.pose.position.y;
  transform_stamped.transform.translation.z = odomAftMapped_.pose.pose.position.z;
  transform_stamped.transform.rotation      = odomAftMapped_.pose.pose.orientation;

  tf_broadcaster_->sendTransform(transform_stamped);
}

void SPARKFastLIO2::publishPath(const state_ikfom &state) {
  setPoseStamp(state, msg_body_pose_, viz_frame_);
  msg_body_pose_.header.stamp    = rclcpp::Time(lidar_end_time_ * 1e9);
  msg_body_pose_.header.frame_id = map_frame_;

  static int jjj = 0;
  jjj++;
  if (jjj % 10 == 0) {
    path_msg_.poses.push_back(msg_body_pose_);
    pub_path_->publish(path_msg_);
  }
}

void SPARKFastLIO2::publishFrameWorld(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloud) {
  if (!scan_pub_en_) {
    return;
  }

  // choose which cloud to publish
  PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en_ ? cloud_undistort_ : feats_down_body_);

  int size = laserCloudFullRes->points.size();
  // allocate world frames
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
  PointCloudXYZI::Ptr laserCloudTmp(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) {
    if (viz_frame_ == "imu") {
      pclPointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
    } else if (viz_frame_ == "lidar") {
      pclPointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudTmp->points[i]);
      pclPointIMUToLiDAR(&laserCloudTmp->points[i], &laserCloudWorld->points[i]);
    } else if (viz_frame_ == "base") {
      pclPointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudTmp->points[i]);
      pclPointIMUToBase(&laserCloudTmp->points[i], &laserCloudWorld->points[i]);
    } else {
      throw std::invalid_argument("Invalid visualization frame has been given");
    }
  }

  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*laserCloudWorld, cloud_msg);
  // use lidar_end_time_ for the timestamp
  cloud_msg.header.stamp    = rclcpp::Time(lidar_end_time_ * 1e9);  // from seconds
  cloud_msg.header.frame_id = map_frame_;

  pubCloud->publish(cloud_msg);
  publish_count_ -= PUBFRAME_PERIOD;

  // Optionally do the pcd_save_en_ part
  if (pcd_save_en_) {
    int nsize = cloud_undistort_->points.size();
    PointCloudXYZI::Ptr laserCloudWorld2(new PointCloudXYZI(nsize, 1));

    for (int i = 0; i < nsize; i++) {
      pclPointBodyToWorld(&cloud_undistort_->points[i], &laserCloudWorld2->points[i]);
    }
    if (pcd_save_interval_ > 0) {
      *cloud_to_be_saved_ += *laserCloudWorld2;  // see below if you store that as a member
    }

    static int scan_wait_num = 0;
    scan_wait_num++;
    if (cloud_to_be_saved_->size() > 0 && pcd_save_interval_ > 0 &&
        scan_wait_num >= pcd_save_interval_) {
      pcd_index_++;
      std::string all_points_dir(std::string(ROOT_DIR) + "PCD/scans_" + std::to_string(pcd_index_) +
                                 ".pcd");
      pcl::PCDWriter pcd_writer;
      std::cout << "Current scan saved to /PCD/ " << all_points_dir << std::endl;
      pcd_writer.writeBinary(all_points_dir, *cloud_to_be_saved_);
      cloud_to_be_saved_->clear();
      scan_wait_num = 0;
    }
  }
}

void SPARKFastLIO2::publishFrame(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloud,
    const std::string &frame) {
  int size = cloud_undistort_->points.size();
  PointCloudXYZI::Ptr laserCloudTransformed(new PointCloudXYZI(size, 1));
  sensor_msgs::msg::PointCloud2 cloud_msg;

  if (frame == "lidar") {
    // direct copy
    for (int i = 0; i < size; i++) {
      laserCloudTransformed->points[i] = cloud_undistort_->points[i];
    }
    pcl::toROSMsg(*laserCloudTransformed, cloud_msg);
    cloud_msg.header.stamp    = rclcpp::Time(lidar_end_time_ * 1e9);
    cloud_msg.header.frame_id = lidar_frame_;
  } else if (frame == "imu") {
    for (int i = 0; i < size; i++) {
      pclPointBodyLidarToIMU(&cloud_undistort_->points[i], &laserCloudTransformed->points[i]);
    }
    pcl::toROSMsg(*laserCloudTransformed, cloud_msg);
    cloud_msg.header.stamp    = rclcpp::Time(lidar_end_time_ * 1e9);
    cloud_msg.header.frame_id = imu_frame_;
  } else if (frame == "base") {
    for (int i = 0; i < size; i++) {
      pclPointBodyLidarToBase(&cloud_undistort_->points[i], &laserCloudTransformed->points[i]);
    }
    pcl::toROSMsg(*laserCloudTransformed, cloud_msg);
    cloud_msg.header.stamp    = rclcpp::Time(lidar_end_time_ * 1e9);
    cloud_msg.header.frame_id = base_frame_;
  } else {
    throw std::invalid_argument("Invalid frame has been given");
  }

  pubCloud->publish(cloud_msg);
  publish_count_ -= PUBFRAME_PERIOD;
}

PoseStruct SPARKFastLIO2::transformPoseWrtLidarFrame(const state_ikfom &state) const {
  // offset_A_B: transformation matrix of A w.r.t. B
  Eigen::Vector3d lidar_position = state.offset_R_L_I.inverse() * (state.rot * state.offset_T_L_I +
                                                                   state.pos - state.offset_T_L_I);

  Eigen::Quaterniond lidar_orientation =
      state.offset_R_L_I.inverse() * state.rot * state.offset_R_L_I;

  PoseStruct output;
  output.position_    = lidar_position;
  output.orientation_ = lidar_orientation;
  return output;
}

void SPARKFastLIO2::main() {
  if (syncPackages(Measures_, verbose_)) {
    processLidarAndImu(Measures_);
  }
}

PoseStruct SPARKFastLIO2::transformPoseWrtBaseFrame(const state_ikfom &state) const {
  static const Eigen::Matrix3d offset_R_B_I = state.offset_R_L_I * lidar_R_wrt_base_.inverse();
  static const Eigen::Vector3d offset_T_B_I =
      -offset_R_B_I * lidar_T_wrt_base_ + state.offset_T_L_I;

  Eigen::Vector3d base_position =
      offset_R_B_I.inverse() * (state.rot * offset_T_B_I + state.pos - offset_T_B_I);

  Eigen::Quaterniond base_orientation =
      Eigen::Quaterniond(offset_R_B_I.inverse() * state.rot * offset_R_B_I);

  PoseStruct output;
  output.position_    = base_position;
  output.orientation_ = base_orientation;
  return output;
}

bool SPARKFastLIO2::syncPackages(MeasureGroup &meas, bool verbose) {
  std::lock_guard<std::mutex> lk(buffer_mutex_);
  if (verbose) {
    static size_t num_lidar_prev = 0;
    static size_t num_imu_prev   = 0;
    size_t num_lidar_curr        = lidar_buffer_.size();
    size_t num_imu_curr          = imu_buffer_.size();

    // To only print out when changes occur
    if ((num_lidar_prev != num_lidar_curr) || (num_imu_prev != num_imu_curr)) {
      RCLCPP_INFO(this->get_logger(), "%lu vs. %lu", num_lidar_curr, num_imu_curr);
      num_lidar_prev = num_lidar_curr;
      num_imu_prev   = num_imu_curr;
    }
  }

  if (lidar_buffer_.empty() || imu_buffer_.empty()) return false;

  if (!lidar_pushed_) {
    meas.lidar                = lidar_buffer_.front();
    meas.lidar_beg_time       = time_buffer_.front();
    static double denominator = 1000;

    if (meas.lidar->points.size() <= 1) {
      lidar_end_time_ = meas.lidar_beg_time + lidar_mean_scantime_;
    } else if (meas.lidar->points.back().curvature / denominator < 0.5 * lidar_mean_scantime_) {
      lidar_end_time_ = meas.lidar_beg_time + lidar_mean_scantime_;
    } else {
      scan_num_++;
      if (meas.lidar->points.back().curvature < 80 || meas.lidar->points.back().curvature > 120) {
        RCLCPP_WARN(this->get_logger(),
                    "meas.lidar->points.back().curvature (%.2f) should be close to 100. Please "
                    "check the `timestamp_unit` "
                    "or values of `time` (or `t`) field of the point cloud input from your sensor.",
                    meas.lidar->points.back().curvature);
      }

      double dt       = meas.lidar->points.back().curvature / 1000.0;
      lidar_end_time_ = meas.lidar_beg_time + dt;
      lidar_mean_scantime_ += (dt - lidar_mean_scantime_) / static_cast<double>(scan_num_);
    }
    meas.lidar_end_time = lidar_end_time_;
    lidar_pushed_       = true;
  }

  if (last_imu_timestamp_.seconds() < lidar_end_time_) {
    if (verbose) {
      static rclcpp::Time last_imu_timestamp_prev;
      // To only print out when changes occur
      if (last_imu_timestamp_prev != last_imu_timestamp_) {
        RCLCPP_INFO(this->get_logger(),
                    "Not enough IMU data (%.6f < %.6f)",
                    last_imu_timestamp_.seconds(),
                    lidar_end_time_);
        last_imu_timestamp_prev = last_imu_timestamp_;
      }
    }
    return false;
  }

  /*** push imu data, and pop from imu buffer ***/
  double imu_time = rclcpp::Time(imu_buffer_.front()->header.stamp).seconds();
  meas.imu.clear();
  while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
    imu_time = rclcpp::Time(imu_buffer_.front()->header.stamp).seconds();
    if (imu_time > lidar_end_time_) break;
    meas.imu.push_back(imu_buffer_.front());
    imu_buffer_.pop_front();
  }

  lidar_buffer_.pop_front();
  time_buffer_.pop_front();
  lidar_pushed_ = false;

  return true;
}

bool SPARKFastLIO2::isMotionStopped(const V3D &acc_ref,
                                    const V3D &acc_curr,
                                    const double acc_diff_thr) {
  return (acc_ref - acc_curr).norm() <= acc_diff_thr;
}

void SPARKFastLIO2::processLidarAndImu(MeasureGroup &Measures) {
  if (flg_first_scan_) {
    first_lidar_time_                = Measures.lidar_beg_time;
    imu_processor_->first_lidar_time = first_lidar_time_;
    flg_first_scan_                  = false;
    return;
  }

  // NOTE(hlim): Place resampling outside `Process` function to get full cloud point,
  // i.e., `cloud_undistort_`: raw undistorted cloud points
  // & `feats_undistort_`: Undistorted cloud points for pose estimation module
  cloud_undistort_->clear();
  feats_undistort_->clear();

  imu_processor_->Process(Measures, kf_, cloud_undistort_);
  feats_undistort_->reserve(cloud_undistort_->size() / point_filter_num_);

  for (size_t i = 0; i < cloud_undistort_->points.size(); i++) {
    if (i % point_filter_num_ == 0) {
      feats_undistort_->push_back(cloud_undistort_->points[i]);
    }
  }

  latest_state_ = kf_.get_x();

  if (feats_undistort_->empty() || (feats_undistort_ == NULL)) {
    RCLCPP_WARN_STREAM(this->get_logger(), "No point, skip this scan!\n");
    return;
  }

  static int num_consecutive_moving_frames = 0;
  if (enable_gravity_alignment_ && !is_gravity_aligned_ && !base_frame_.empty()) {
    if (!flg_EKF_inited_) {
      // Assume that it is stationary at the beginning.
      mean_acc_stopped_ = Measures.getMeanAcc();
    } else {
      const auto &mean_acc = Measures.getMeanAcc();
      if (isMotionStopped(mean_acc_stopped_, mean_acc, acc_diff_thr_)) {
        RCLCPP_WARN_STREAM(
            this->get_logger(),
            "Waiting for motion to perform gravity alignment...now a robot has been stopped");
        num_consecutive_moving_frames = 0;
      } else {
        num_consecutive_moving_frames = min(num_consecutive_moving_frames + 1, 100000);
      }
    }
  }

  flg_EKF_inited_ = (Measures.lidar_beg_time - first_lidar_time_) < INIT_TIME ? false : true;
  lasermapFovSegment();

  down_size_filter_.setInputCloud(feats_undistort_);
  down_size_filter_.filter(*feats_down_body_);
  feats_down_size_ = feats_down_body_->points.size();

  if (ikd_tree_.Root_Node == nullptr) {
    if (feats_down_size_ > 5) {
      ikd_tree_.set_downsample_param(filter_size_map_min_);
      feats_down_world_->resize(feats_down_size_);
      for (int i = 0; i < feats_down_size_; i++) {
        pointBodyToWorld(
            &(feats_down_body_->points[i]), &(feats_down_world_->points[i]), latest_state_);
      }
      ikd_tree_.Build(feats_down_world_->points);
    }
    return;
  }
  kdtree_size_st_ = ikd_tree_.size();

  /*** ICP and iterated Kalman filter update ***/
  if (feats_down_size_ < 5) {
    RCLCPP_WARN_STREAM(this->get_logger(), "No point, skip this scan!");
    return;
  }

  normvec_->resize(feats_down_size_);
  feats_down_world_->resize(feats_down_size_);

  nearest_points_.resize(feats_down_size_);

  /*** iterated state estimation ***/
  double t_update_start = omp_get_wtime();
  double solve_H_time   = 0;
  kf_.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);

  /***** Perform gravity alignment *****/
  // NOTE(hlim): Introduce a delay using `num_moving_frames_thr_` to make sure
  // the gravity vectors are sufficiently updated.
  if (enable_gravity_alignment_ && !is_gravity_aligned_ && !base_frame_.empty() &&
      (num_consecutive_moving_frames > num_moving_frames_thr_)) {
    static const auto &offset_R_I_B = lidar_R_wrt_base_ * latest_state_.offset_R_L_I.inverse();

    // NOTE(hlim): Here, we don't need to normalize the scale of vectors
    V3D gravity_direction = kf_.get_x().grav;
    if (global_gravity_directions_.size() < static_cast<size_t>(num_gravity_measurements_thr_)) {
      {
        std::stringstream ss;
        ss << "Waiting for motion: " << global_gravity_directions_.size() << " / "
           << num_gravity_measurements_thr_;
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
      }

      global_gravity_directions_.push_back(offset_R_I_B * gravity_direction);
    } else {
      V3D avg_global_gravity_vec = Eigen::Vector3d::Zero();
      for (const auto &gravity_vec : global_gravity_directions_) {
        avg_global_gravity_vec += gravity_vec;
      }
      avg_global_gravity_vec /= global_gravity_directions_.size();

      R_gravity_aligned_ = computeRelativeRotation(avg_global_gravity_vec, g_base_);

      {
        std::stringstream ss;
        ss << "Gravity alignment complete! `R_gravity_aligned`: " << R_gravity_aligned_;
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
      }

      is_gravity_aligned_ = true;
    }
  }

  latest_state_          = kf_.get_x();
  kf_for_preintegration_ = kf_;
  // Update corrected rotation here
  latest_state_.pos = R_gravity_aligned_ * latest_state_.pos;
  latest_state_.rot = R_gravity_aligned_ * latest_state_.rot;

  if (enable_gravity_alignment_ && !is_gravity_aligned_ && !base_frame_.empty()) {
    RCLCPP_WARN(this->get_logger(),
                "Gravity alignment is enabled but not yet completed. Waiting for alignment...");
    return;
  }

  /******* Publish topics *******/
  const auto stamp = rclcpp::Time(lidar_end_time_ * 1e9);
  publishOdometry(latest_state_, stamp);
  mapIncremental();

  if (path_en_) {
    publishPath(latest_state_);
  }
  if (scan_pub_en_) {
    publishFrameWorld(pub_cloud_full_);
    if (scan_lidar_pub_en_) publishFrame(pub_cloud_lidar_, "lidar");
    if (scan_body_pub_en_) publishFrame(pub_cloud_body_, "imu");
    if (scan_base_pub_en_) publishFrame(pub_cloud_base_, "base");
  }
}
}  // namespace spark_fast_lio

RCLCPP_COMPONENTS_REGISTER_NODE(spark_fast_lio::SPARKFastLIO2)
