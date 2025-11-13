# VelGrasp
This repository contains the framework, mathematics and codes for performing Dynamic task by two heterogeneos robots using vrlocity control. 

A second order dynamical system has been used to throw the object using two robots. 

DS would give the desired object acceleration to throw the object. The computed object acceleration would go as a feedforward term to the object level impedance controlequation(refer my ICRA paper(equation 13 or 14)).

######################### Loading the model ############################

One can load the model of two heal robtos with a box in between using **test.py** file available in the **scripts** folder of the repository.
XML required to load the model is **scene.xml** which comprises of botht he robtos wiht the box in middle and the available sites to grasp the box from its sides.

In the **assets** folder inside **robot_description**, one can find the availibale xml of a sigle heal robot.

############# Inverse Kinematics (IK) to make the robots reach the grasp sites of the box #################

In the **mink IK** folder inside the the **scripts**, one can find the dual and single arm IK codes, where the robots reache the target sites of the box following a pre grasp site.


######### USER CAN WRITE HIS CODES FOR THE **DYNAMIC TASK** INSIDE THE SCRIPTS FOLDER #########



