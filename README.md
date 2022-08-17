Code to integrate https://github.com/lewjiayi/Crowd-Analysis with the NIST-ACE framework: https://github.com/usnistgov/ace

```
# Build the image
./build.sh

# Run
CONTAINER_ID="crowd-analysis:latest" ./runDocker.sh

# In container, test xeyes
xeyes
# should display

# code is automatically mounted in /dmc/code
cd /dmc/code

#run without connecting to ACE
./run.sh

#to run with ace (need to edit registar_analytic() to match host ip):
python3 rpc_analytic.py

```
