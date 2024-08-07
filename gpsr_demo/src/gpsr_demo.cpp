#include "behaviortree_cpp_v3/behavior_tree.h"
#include "behaviortree_cpp_v3/bt_factory.h"
#include "behaviortree_cpp_v3/loggers/bt_zmq_publisher.h"
#include "behaviortree_cpp_v3/utils/shared_library.h"

#include "ament_index_cpp/get_package_share_directory.hpp"

#include "gpsr_msgs/srv/generate_plan.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <string>

using namespace std::chrono_literals;

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("gpsr_demo_node");
  BT::BehaviorTreeFactory factory;
  BT::SharedLibrary loader;
  bool execution = true;

  rclcpp::Client<gpsr_msgs::srv::GeneratePlan>::SharedPtr client =
      node->create_client<gpsr_msgs::srv::GeneratePlan>("gpsr_planning");

  std::list<std::string> commands = {
      "Tell me how many people in the livingroom are wearing orange shirts.",
      "Tell me how many snacks there are in the coffee table",
      "Say the day of the week to the person pointing to the left in the bedroom",
      "Say what day is tomorrow to the waving person in the kitchen",
            "Tell me the gesture of the person in the livingroom",
      "Say your teams name to the person raising their right arm in the bedroom",
      "Say your teams country to the person raising their left arm in the living room",
      "Say what day is today to the waving person in the bedroom",
      "Meet Unique in the office and tell your teams affiliation",
      "Say hello to James in the bedroom and tell what day is tomorrow",
      "Tell the time to the waving person in the corridor",
      "Introduce yourself to Kyle in the livingroom and guide them to the kitchen",
      "Look for a drink in the kitchen, then get it and deliver it to Kin in the kitchen.",
      "Get a dice from the dishwasher and give it to the person rising their left arm in the bedroom.",
      "Locate a bottle in the kitchen, then fetch it and place it on the bed.",
      "Answer the quiz of the waving person in the bedroom",
      "Follow the standing person in the kitchen",
      "Follow Ana from the kitchen to the bedroom",
      "Follow the standing person at the kitchen",
      "Navigate to the kitchen then find a dish and take it and give it to me",
      "Salute the person wearing an orange t shirt in the dining room and "
      "follow them to the living room",
      "Escort the person wearing a yellow blouse from the kitchen to the "
      "kitchen",
      "Navigate to the kitchen then look for a fruit and take it and give it "
      "to me",
      "Get a spoon from the living room and put it on the bedroom",
      "Go to the living room then look for a fruit and take it and bring it to "
      "Juan in the kitchen",
      "Locate a plate in the dining room then get it and deliver it to me",
      "Lead the person wearing a gray blouse from the kitchen to the kitchen",
      "Navigate to the living room then locate a fruit and get it and give it "
      "to Luis in the bedroom",
      "Tell me what is the thinnest fruit on the bedroom",
      "Tell me how many persons pointing to the left are in the kitchen",
      "Fetch a dish from the living room and put it on the bedroom",
      "Tell something about yourself to the person pointing to the right in "
      "the dining room",
      "Answer the quiz of the person pointing to the right in the kitchen",
      "Lead the lying person from the kitchen to the kitchen"};

  std::list<std::string>::iterator it = commands.begin();
  factory.registerFromPlugin(loader.getOSName("follow_person_bt_node"));
  factory.registerFromPlugin(loader.getOSName("guide_person_bt_node"));
  factory.registerFromPlugin(loader.getOSName("count_bt_node"));
  factory.registerFromPlugin(loader.getOSName("describe_object_bt_node"));
  factory.registerFromPlugin(loader.getOSName("describe_person_bt_node"));
  factory.registerFromPlugin(loader.getOSName("find_object_bt_node"));
  factory.registerFromPlugin(loader.getOSName("find_person_bt_node"));
  factory.registerFromPlugin(loader.getOSName("listen_bt_node"));
  factory.registerFromPlugin(loader.getOSName("look_around_bt_node"));
  factory.registerFromPlugin(loader.getOSName("look_to_bt_node"));
  factory.registerFromPlugin(loader.getOSName("move_to_bt_node"));
  factory.registerFromPlugin(loader.getOSName("offer_bt_node"));
  factory.registerFromPlugin(loader.getOSName("pick_object_bt_node"));
  factory.registerFromPlugin(loader.getOSName("place_object_bt_node"));
  factory.registerFromPlugin(loader.getOSName("query_bt_node"));
  factory.registerFromPlugin(loader.getOSName("recognize_person_bt_node"));
  factory.registerFromPlugin(loader.getOSName("speak_bt_node"));
  std::string pkgpath =
      ament_index_cpp::get_package_share_directory("gpsr_demo");
  std::string xml_file = pkgpath + "/bt_xml/gpsr_demo.xml";

  std::cout << "Press ENTER to execute next command or 'q' to quit."
            << std::endl;

  while (execution) {
    std::string input;
    std::getline(std::cin, input);

    if (input != "q") {
      if (it != commands.end()) {
        std::string command = *it;
        std::cout << "Command: " << command.c_str() << std::endl;
        it = std::next(it, 1);

        auto request =
            std::make_shared<gpsr_msgs::srv::GeneratePlan::Request>();
        request->command = command;

        while (!client->wait_for_service(1s)) {
          if (!rclcpp::ok()) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"),
                         "Interrupted while waiting for the service. Exiting.");
            return 0;
          }
          RCLCPP_INFO(rclcpp::get_logger("rclcpp"),
                      "service not available, waiting again...");
        }

        auto result = client->async_send_request(request);

        // Wait for the result.
        if (rclcpp::spin_until_future_complete(node, result) ==
            rclcpp::FutureReturnCode::SUCCESS) {
          RCLCPP_INFO(rclcpp::get_logger("rclcpp"),
                      "Success service execute_plan");
          std::cout << "Success service execute_plan " << std::endl;
        } else {
          RCLCPP_ERROR(rclcpp::get_logger("rclcpp"),
                       "Failed to call service execute_plan");
        }

        auto blackboard = BT::Blackboard::create();
        blackboard->set("node", node);

        // BT::Tree tree = factory.createTreeFromText(result->bt_xml);

        // auto publisher_zmq =
        //     std::make_shared<BT::PublisherZMQ>(tree, 10, 2666, 2667);

        // rclcpp::Rate rate(10);

        // BT::NodeStatus status = BT::NodeStatus::RUNNING;
        // bool finish = false;
        // while (!finish && rclcpp::ok()) {
        //   status = tree.rootNode()->executeTick();
        //   finish = (status == BT::NodeStatus::SUCCESS) ||
        //            (status == BT::NodeStatus::FAILURE);

        //   rclcpp::spin_some(node);
        //   rate.sleep();
        // }
        // std::cout << "GPSR demo Finished with status: " << status <<
        // std::endl; std::cout << "Press ENTER to execute next command or 'q'
        // to quit."
        //           << std::endl;

      } else {
        it = commands.begin();
      }
    } else {
      execution = false;
    }
  }

  rclcpp::shutdown();
  return 0;
}