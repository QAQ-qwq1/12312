[app]
title = RUJIJING
package.name = xmasrose
package.domain = com.particle
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1

# 【关键】依赖包，必须包含 numpy 和 pyopengl
requirements = python3,kivy,numpy,pyopengl

# 权限
android.permissions = INTERNET

# 架构 (支持更多手机)
android.archs = arm64-v8a, armeabi-v7a

# 屏幕方向 (横屏看3D效果好)
orientation = portrait

# 全屏
fullscreen = 1

# Android API 设置
android.api = 31
android.minapi = 21

# --- 以下保持默认即可，为了精简省略了其他默认注释 ---
[buildozer]
log_level = 2
warn_on_root = 1