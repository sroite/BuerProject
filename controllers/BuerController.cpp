#include <cnoid/SimpleController>
#include <cnoid/Body>
#include <cnoid/Link>
#include <vector>
#include <string>

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

        const std::vector<std::string> leg_names = {"leg1", "leg2", "leg3", "leg4", "leg5"};
        const std::vector<std::string> joint_types = {"_lap", "_calf", "_foot"};

        io->os() << "Initializing BuerHexapodController..." << std::endl;

        for (const auto& leg_name : leg_names) {
            for (const auto& joint_type : joint_types) {
                std::string joint_name = leg_name + joint_type;
                Link* joint = body->link(joint_name);

                if (joint) {
                    legJoints.push_back(joint);
                } else {
                    io->os() << "Error: Joint " << joint_name << " not found in the model!" << std::endl;
                    return false; // Initialization failed
                }
            }
        }

        int numJoints = legJoints.size();
        q_ref.resize(numJoints);
        q_prev.resize(numJoints);

        for (int i = 0; i < numJoints; ++i) {
            Link* joint = legJoints[i];
            joint->setActuationMode(Link::JointTorque);
            io->enableIO(joint);
            q_ref[i] = q_prev[i] = joint->q(); // joint->q() 返回当前关节角度（弧度）
            io->os() << "Joint " << joint->name() << " initialized with initial angle: " << q_ref[i] << " radians." << std::endl;
        }

        dt = io->timeStep();

        io->os() << "BuerHexapodController initialized successfully. Controlling " << numJoints << " joints." << std::endl;

        return true;
    }

    // 控制方法，在每个模拟步长中调用
    virtual bool control() override
    {
        int numJoints = legJoints.size();
        for (int i = 0; i < numJoints; ++i) {
            Link* joint = legJoints[i];
            double q = joint->q();
            double dq = (q - q_prev[i]) / dt;
            double dq_ref = 0.0;
            double torque = P_gain * (q_ref[i] - q) + D_gain * (dq_ref - dq);
            joint->u() = torque; // joint->u() 是关节的力矩或力输入
            q_prev[i] = q;
        }

        return true;
    }
};
CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(BuerHexapodController)