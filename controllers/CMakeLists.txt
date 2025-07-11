# 找到 LibTorch
find_package(Torch REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

include_directories(${Torch_INCLUDE_DIRS})
include_directories(
  ${Torch_INCLUDE_DIRS}
  ${Qt5Widgets_INCLUDE_DIRS}
)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# 构建BuerInferenceController
# 假设源文件路径是 buer/src/controllers/BuerInferenceController.cpp
# 注意：路径是相对于 src 目录的
choreonoid_add_simple_controller(BuerInferenceController BuerInferenceController.cpp)

# 链接需要的库
target_link_libraries(BuerInferenceController
  ${CHOREONOID_LIBRARIES}
  "${TORCH_LIBRARIES}"
  ${Qt5Widgets_LIBRARIES}
)

# 构建Buer_Controller
# 假设源文件路径是 buer/src/BuerController.cpp
choreonoid_add_simple_controller(Buer_Controller BuerController.cpp)
target_link_libraries(Buer_Controller
  ${CHOREONOID_LIBRARIES}
  ${Qt5Widgets_LIBRARIES}
  "${TORCH_LIBRARIES}"
)
