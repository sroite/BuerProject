#include <cnoid/SimpleController>
#include <cnoid/Body>
#include <cnoid/Link>
#include <vector>
#include <string>
#include <iostream>

using namespace cnoid;

class BuerHexapodController : public SimpleController
{
    std::vector<Link*> legJoints;
    std::vector<double> q_ref;
    std::vector<double> q_prev;
    double dt;

    const double P_gain = 500.0;
    const double D_gain = 50.0;

public:
    virtual bool initialize(SimpleControllerIO* io) override
    {
        Body* body = io->body();

        io->os() << "=== All links in body ===" << std::endl;
        for(auto& l : body->links()){
            io->os() << l->name() << std::endl;  // 打印模型里所有 link 名称
        }
        io->os() << "========================" << std::endl;

        const std::vector<std::string> leg_names = {"leg1", "leg2", "leg3", "leg4"};
        const std::vector<std::string> joint_types = {"_lap", "_calf", "_foot"};

        io->os() << "Initializing BuerHexapodController..." << std::endl;

        for (const auto& leg_name : leg_names) {
            for (const auto& joint_type : joint_types) {
                std::string joint_name = leg_name + joint_type; // 这里是 link 的名字
                Link* joint = body->link(joint_name);

                if (joint) {
                    legJoints.push_back(joint);
                    io->os() << "Loaded joint: " << joint_name << std::endl;
                } else {
                    io->os() << "Warning: Joint " << joint_name << " not found in the model!" << std::endl;
                    // 不返回 false，继续加载其他 joints
                }
            }
        }

        int numJoints = legJoints.size();
        if(numJoints == 0){
            io->os() << "Error: No joints loaded. Controller will do nothing." << std::endl;
            return false;
        }

        q_ref.resize(numJoints);
        q_prev.resize(numJoints);

        for (int i = 0; i < numJoints; ++i) {
            Link* joint = legJoints[i];
            joint->setActuationMode(Link::JointTorque);
            io->enableIO(joint);
            q_ref[i] = q_prev[i] = joint->q();
            io->os() << "Joint " << joint->name() << " initialized with angle: " << q_ref[i] << std::endl;
        }

        dt = io->timeStep();
        io->os() << "Controller initialized successfully. Total joints: " << numJoints << std::endl;

        return true;
    }

    virtual bool control() override
    {
        if(legJoints.empty()) return true;  // 如果没有 joints，就什么也不做

        for (size_t i = 0; i < legJoints.size(); ++i) {
            Link* joint = legJoints[i];
            double q = joint->q();
            double dq = (q - q_prev[i]) / dt;
            double dq_ref = 0.0;
            double torque = P_gain * (q_ref[i] - q) + D_gain * (dq_ref - dq);
            joint->u() = torque;
            q_prev[i] = q;
        }

        return true;
    }
};

CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(BuerHexapodController)
