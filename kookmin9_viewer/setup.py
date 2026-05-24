from setuptools import find_packages, setup

package_name = 'kookmin9_viewer'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lhcho',
    maintainer_email='lhcho@todo.todo',
    description='Xytron Unity 시뮬레이터 토픽 시각화 + 차량 조종 테스트 노드',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'test_viewer = kookmin9_viewer.test_viewer:main',
        ],
    },
)
