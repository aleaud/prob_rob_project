PYTHON = python3

.PHONY = test-traj test-env tls-pc tls-triang clean

FILES = pr_classes.py utils.py estimates.py errors_jacobians.py pr_cast.py tls.py constants.py pr_plot.py
TEST_FILES = pr_classes.py constants.py

test-traj: 
	${PYTHON} test_trajectory.py ${TEST_FILES}

test-env: 
	${PYTHON} test_env.py ${TEST_FILES}

test-est:
	${PYTHON} test_est.py ${TEST_FILES}

tls-pc:
	${PYTHON} main_pc.py ${FILES}

tls-triang: 
	${PYTHON} main_triang.py ${FILES}

clean:
	rm -rf __pycache__