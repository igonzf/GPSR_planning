// Copyright 2024 Intelligent Robotics Lab - Gentlebots
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BT_NODES__CONVERT_COLOR_HPP_
#define BT_NODES__CONVERT_COLOR_HPP_

#include <algorithm>
#include <string>

#include "behaviortree_cpp_v3/behavior_tree.h"
#include "behaviortree_cpp_v3/bt_factory.h"
#include "rclcpp_cascade_lifecycle/rclcpp_cascade_lifecycle.hpp"

namespace bt_nodes {

class ConvertColor : public BT::ActionNodeBase {
public:
  explicit ConvertColor(const std::string &xml_tag_name,
                        const BT::NodeConfiguration &conf);

  void halt();
  BT::NodeStatus tick();

  static BT::PortsList providedPorts() { return BT::PortsList({}); }

private:
  rclcpp_cascade_lifecycle::CascadeLifecycleNode::SharedPtr node_;

};

} // namespace bt_nodes

#endif // BT_NODES__CONVERT_COLOR_HPP_
