from setuptools import find_packages, setup

package_name = 'status_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='imjunsik',
    maintainer_email='imjunsik@todo.todo',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ip_publisher = status_pkg.ip_publisher:main',
            'ble_provision_server = status_pkg.ble_provision_server:main',
            'status_service = status_pkg.status_servic_node:main'
        ],
    },
)
