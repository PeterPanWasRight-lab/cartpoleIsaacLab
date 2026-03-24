# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
## import_packages 会自动扫描当前包目录下的所有子模块（除了黑名单中的 utils 和 .mdp）
# 并递归导入它们的 __init__.py 文件，从而触发这些子模块中的 gym.register() 调用
# 最终将所有自定义的 Gym 环境注册到 gymnasium 的注册表中，使其可以通过 gym.make() 创建
