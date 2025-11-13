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


#### USER CAN WRITE HIS CODES FOR THE **DYNAMIC TASK** INSIDE THE SCRIPTS FOLDER ####




&&&&&&&&&&&&&&&& IMPORTANT $$$$$$$$$$$$$$$$$$$$$$$$
# 1. Clone the repository
git clone git@github.com:Samriddhi-dubey06/samriddhi_mujoco_bimanual_ds_dynamic_task.git

# 2. Enter the project directory
cd samriddhi_mujoco_bimanual_ds_dynamic_task

# 3. Fetch all branches
git fetch origin

# 4. Switch to the dev branch (IMPORTANT: do NOT use main)
git checkout dev

# 5. Make your changes in the code
# ---- edit files normally ----

# 6. Stage your changes
git add .

# 7. Commit your changes
git commit -m "Describe what you changed"

# 8. Push to the dev branch
git push origin dev

# (If first time pushing)
git push -u origin dev




