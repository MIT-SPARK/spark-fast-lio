#pragma once

#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include "common/so3_math.h"
#include "ikd_Tree.h"
#include "imu_processing.hpp"
#include "preprocess.h"

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)
#define MOV_THRESHOLD (1.5)

#if defined(LIVOX_ROS_DRIVER_FOUND) && LIVOX_ROS_DRIVER_FOUND
#include <livox_ros_driver2/msg/custom_msg.hpp>
#endif

namespace spark_fast_lio {

struct PoseStruct {
  Eigen::Vector3d position_;
  Eigen::Quaterniond orientation_;
};

class SPARKFastLIO2 : public rclcpp::Node {
 public:
  explicit SPARKFastLIO2(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

 private:
  M3D computeRelativeRotation(const Eigen::Vector3d &g_a, const Eigen::Vector3d &g_b);

  bool lookupBaseExtrinsics(V3D &lidar_T_wrt_base, M3D &lidar_R_wrt_base);

  void pointBodyToWorld(PointType const *const pi, PointType *const po, const state_ikfom &s);

  template <typename T>
  void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi,
                        Eigen::Matrix<T, 3, 1> &po,
                        const state_ikfom &s) const {
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
  }

  void pclPointBodyToWorld(PointType const *const pi, PointType *const po);

  void pclPointBodyLidarToIMU(PointType const *const pi, PointType *const po);

  void pclPointBodyLidarToBase(PointType const *const pi, PointType *const po);

  void pclPointIMUToLiDAR(PointType const *const pi, PointType *const po);

  void pclPointIMUToBase(PointType const *const pi, PointType *const po);

  void collectRemovedPoints();

  void standardLiDARCallback(const sensor_msgs::msg::PointCloud2 &msg);

#if defined(LIVOX_ROS_DRIVER_FOUND) && LIVOX_ROS_DRIVER_FOUND
  void livoxLiDARCallback(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr msg);
#endif

  void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg);

  void integrateIMU(esekfom::esekf<state_ikfom, 12, input_ikfom> &state,
                    const sensor_msgs::msg::Imu &msg);

  void calcHModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);

  void lasermapFovSegment();

  void mapIncremental();

  void publishOdometry(const state_ikfom &state, const rclcpp::Time &stamp);

  void publishPath(const state_ikfom &state);

  void publishFrameWorld(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloud);

  void publishFrame(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloud,
                    const std::string &frame);

  PoseStruct transformPoseWrtLidarFrame(const state_ikfom &state) const;

  PoseStruct transformPoseWrtBaseFrame(const state_ikfom &state) const;

  template <typename T>
  void setPoseStamp(const state_ikfom &state, T &out, const std::string &frame) const {
    if (frame == "imu") {
      out.pose.position.x    = state.pos(0);
      out.pose.position.y    = state.pos(1);
      out.pose.position.z    = state.pos(2);
      const auto quat        = state.rot.coeffs();
      out.pose.orientation.x = quat[0];
      out.pose.orientation.y = quat[1];
      out.pose.orientation.z = quat[2];
      out.pose.orientation.w = quat[3];
    } else if (frame == "lidar") {
      const auto &p          = transformPoseWrtLidarFrame(state);
      out.pose.position.x    = p.position_(0);
      out.pose.position.y    = p.position_(1);
      out.pose.position.z    = p.position_(2);
      out.pose.orientation.x = p.orientation_.x();
      out.pose.orientation.y = p.orientation_.y();
      out.pose.orientation.z = p.orientation_.z();
      out.pose.orientation.w = p.orientation_.w();
    } else if (frame == "base") {
      const auto &p          = transformPoseWrtBaseFrame(state);
      out.pose.position.x    = p.position_(0);
      out.pose.position.y    = p.position_(1);
      out.pose.position.z    = p.position_(2);
      out.pose.orientation.x = p.orientation_.x();
      out.pose.orientation.y = p.orientation_.y();
      out.pose.orientation.z = p.orientation_.z();
      out.pose.orientation.w = p.orientation_.w();
    } else {
      throw std::invalid_argument("Invalid visualization frame has been given");
    }
  }

  void main();

  bool syncPackages(MeasureGroup &meas, bool verbose);

  bool isMotionStopped(const V3D &acc_ref, const V3D &acc_curr, const double acc_diff_thr);

  void processLidarAndImu(MeasureGroup &Measure);
 private:
  std::mutex buffer_mutex_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_;

#if defined(LIVOX_ROS_DRIVER_FOUND) && LIVOX_ROS_DRIVER_FOUND
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_lidar_livox_;
#endif

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_full_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_lidar_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_body_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_base_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Clock::SharedPtr clock_;
  rclcpp::TimerBase::SharedPtr main_loop_timer_;

  /*** Time Log Variables ***/
  double kdtree_incremental_time_ = 0.0;
  double kdtree_search_time_      = 0.0;
  double kdtree_delete_time_      = 0.0;

  double match_time_         = 0.0;
  double solve_time_         = 0.0;
  double solve_const_H_time_ = 0.0;

  std::array<double, MAXN> T1_;
  std::array<double, MAXN> s_plot_;
  std::array<double, MAXN> s_plot2_;
  std::array<double, MAXN> s_plot3_;
  std::array<double, MAXN> s_plot4_;
  std::array<double, MAXN> s_plot5_;
  std::array<double, MAXN> s_plot6_;
  std::array<double, MAXN> s_plot7_;
  std::array<double, MAXN> s_plot8_;
  std::array<double, MAXN> s_plot9_;
  std::array<double, MAXN> s_plot10_;
  std::array<double, MAXN> s_plot11_;

  bool runtime_pos_log_ = false;
  /**************************/
  int kdtree_size_st_        = 0;
  int kdtree_size_end_       = 0;
  int add_point_size_        = 0;
  int kdtree_delete_counter_ = 0;

  bool pcd_save_en_       = false;
  bool time_sync_en_      = false;
  bool extrinsic_est_en_  = false;
  bool path_en_           = true;
  bool scan_pub_en_       = false;
  bool dense_pub_en_      = false;
  bool scan_lidar_pub_en_ = false;
  bool scan_body_pub_en_  = false;
  bool scan_base_pub_en_  = false;

  bool verbose_ = false;
  bool pcl_verbose_ = true;

  bool enable_gravity_alignment_ = false;
  bool is_gravity_aligned_       = false;

  float res_last_[100000] = {0.0};
  float det_range_        = 300.0f;

  std::mutex mtx_buffer_;
  std::condition_variable sig_buffer_;

  std::string root_dir_ = ROOT_DIR;
  std::string map_file_path_;
  std::string save_dir_;
  std::string sequence_name_;
  std::string map_frame_;
  std::string lidar_frame_;
  std::string base_frame_;
  std::string imu_frame_;
  std::string viz_frame_;

  double res_mean_last_          = 0.05;
  double total_residual_         = 0.0;
  rclcpp::Time last_lidar_timestamp_;
  rclcpp::Time last_imu_timestamp_;
  int64_t timediff_lidar_wrt_imu_ = 0;

  double gyr_cov_   = 0.1;
  double acc_cov_   = 0.1;
  double b_gyr_cov_ = 0.0001;
  double b_acc_cov_ = 0.0001;

  double filter_size_map_smaller_ = 0.0;
  double filter_size_map_min_     = 0.0;
  double fov_deg_                 = 0.0;
  double cube_len_                = 0.0;
  double total_distance_          = 0.0;
  double lidar_end_time_          = 0.0;
  double first_lidar_time_        = 0.0;

  int effect_feat_num_  = 0;
  int time_log_counter_ = 0;
  int scan_count_       = 0;
  int publish_count_    = 0;

  int iterCount_                = 0;
  int feats_down_size_          = 0;
  int NUM_MAX_ITERATIONS_       = 0;
  int laserCloudValidNum_       = 0;
  int pcd_save_interval_        = -1;
  int pcd_index_                = 0;
  int point_filter_num_         = 4;  // empirically, 4 showed the best performance
  int feats_down_size_neighbor_ = numeric_limits<int>::max();

  double lidar_mean_scantime_ = 0.0;
  int scan_num_               = 0;

  double acc_diff_thr_              = 0.2;
  int num_moving_frames_thr_        = 10;
  int num_gravity_measurements_thr_ = 10;
  bool point_selected_surf_[100000] = {0};
  bool lidar_pushed_                = false;
  bool flg_first_scan_              = true;
  bool flg_exit_                    = false;
  bool flg_EKF_inited_;

  std::deque<V3D> global_gravity_directions_;

  BoxPointType localmap_points_;
  std::vector<BoxPointType> cub_needrm_;

  std::vector<PointVector> nearest_points_;
  std::vector<double> g_base_vec_{0.0, 0.0, -1.0};
  std::vector<double> extrinT_{0.0, 0.0, 0.0};
  std::vector<double> extrinR_{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  double extrinsics_timeout_s_ = 10.0;

  std::deque<double> time_buffer_;
  std::deque<PointCloudXYZI::Ptr> lidar_buffer_;
  std::deque<std::shared_ptr<const sensor_msgs::msg::Imu>> imu_buffer_;

  PointCloudXYZI::Ptr cloud_undistort_;
  PointCloudXYZI::Ptr feats_undistort_;
  PointCloudXYZI::Ptr feats_down_body_;
  PointCloudXYZI::Ptr feats_down_world_;
  PointCloudXYZI::Ptr surface_normals_;
  PointCloudXYZI::Ptr normvec_;
  PointCloudXYZI::Ptr laser_cloud_ori_;
  PointCloudXYZI::Ptr corr_normvec_;
  PointCloudXYZI::Ptr cloud_to_be_saved_;

  pcl::VoxelGrid<PointType> down_size_filter_;
  KD_TREE<PointType> ikd_tree_;

  V3F xaxis_point_body_;
  V3F xaxis_point_world_;
  V3D g_base_;
  V3D mean_acc_stopped_;
  V3D position_last_;
  V3D lidar_T_wrt_imu_;
  M3D lidar_R_wrt_imu_;
  M3D R_gravity_aligned_;

  /*** Only used for integration with the Hydra system ***/
  V3D lidar_T_wrt_base_;
  M3D lidar_R_wrt_base_;

  /*** EKF inputs and output ***/
  MeasureGroup Measures_;
  esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;
  std::optional<esekfom::esekf<state_ikfom, 12, input_ikfom>> kf_for_preintegration_;
  state_ikfom latest_state_;

  nav_msgs::msg::Path path_msg_;
  nav_msgs::msg::Odometry odomAftMapped_;
  geometry_msgs::msg::PoseStamped msg_body_pose_;

  std::shared_ptr<Preprocess> preprocessor_;
  std::shared_ptr<ImuProcess> imu_processor_;
};

}  // namespace spark_fast_lio
