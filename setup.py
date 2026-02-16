from setuptools import find_packages, setup

package_name = "safe_flow_mpc"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    install_requires=["setuptools"],
    include_package_data=True,
    package_data={
        package_name: ["*"],
    },
    maintainer="Thies Oelerich",
    maintainer_email="thies.oelerich@tuwien.ac.at",
    description="SafeFlowMPC Package",
    license="MIT",
)
