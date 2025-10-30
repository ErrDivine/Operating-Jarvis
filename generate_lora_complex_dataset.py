from google import genai
from google.genai import types
import json
import time

# -------------------------- 1. 基础配置 --------------------------
API_KEY = "sk-USul5hvMppKKq7updrxOCPsR7MGHXuDsGa1JPYJbCZbOVVu5"
BASE_URL = "https://turingai.plus"

# 使用您提供的实际工具定义
tools_definition = [

        {
            "type": "function",
            "function": {
                "name": "powerDevice",
                "description": "重启/关闭/锁屏/休眠设备",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Reboot": {
                            "type": "Boolean",
                            "description": "用于识别用户想要重启设备的语音指令。该意图能够理解用户通过自然语言表达的设备重启需求，包括重启手机、重新启动、系统重启、关机重启等相关操作指令。"
                        },
                        "ShutDown": {
                            "type": "Boolean",
                            "description": "用于识别用户想要关闭设备的语音指令。该意图能够理解用户通过自然语言表达的设备关机需求，包括关机、关闭手机、系统关机、电源关闭等相关操作指令。"
                        },
                        "Sleep": {
                            "type": "Boolean",
                            "description": "用于识别用户想要让设备进入睡眠或待机状态的语音指令。该意图能够理解用户通过自然语言表达的设备休眠需求，包括进入睡眠模式、待机、休眠、锁屏待机等相关操作指令。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "placeCall",
                "description": "拨打/接听电话并查询联系人",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Call": {
                            "type": "Boolean",
                            "description": "用于识别用户想要拨打电话的语音指令。该意图能够理解用户通过自然语言表达的拨打电话需求，包括拨打给指定联系人、拨打特定号码、选择手机卡、设置通话模式等操作。"
                        },
                        "CheckContact": {
                            "type": "Boolean",
                            "description": "用于识别用户想要查看或搜索联系人信息的语音指令。该意图能够理解用户通过自然语言表达的联系人查询需求，包括查找特定联系人、查看联系人详细信息、搜索联系人列表、查询联系人的电话号码或其他信息等操作。"
                        },
                        "AnswerCall": {
                            "type": "Boolean",
                            "description": "用于识别用户想要接听电话的语音指令。该意图能够理解用户通过自然语言表达的接听电话需求，包括直接接听来电、使用免提模式接听、拒绝接听或挂断电话等操作。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "runSystemUtility",
                "description": "创建便签/备忘录",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "CreateNote": {
                            "type": "Boolean",
                            "description": "用于识别用户想要创建笔记或备忘录的语音指令。该意图能够理解用户通过自然语言表达的笔记创建需求，包括在指定应用中创建笔记、设置笔记标题和内容、指定笔记类型以及设置提醒时间等操作。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "setVolume",
                "description": "调节媒体/通话/闹钟音量",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ControlSound": {
                            "type": "Boolean",
                            "description": "用于识别用户想要控制设备音量的语音指令。该意图能够理解用户通过自然语言表达的音量调节需求，包括调节系统音量、媒体音量、通话音量、闹钟音量等不同类型的音量控制操作指令。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "setFont",
                "description": "查看并调整字体、字号与粗细",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "CheckFont": {
                            "type": "Boolean",
                            "description": "用于识别用户想要查看或检查字体设置的语音指令。该意图能够理解用户通过自然语言表达的字体查询需求，包括查看当前字体、字体大小、字体样式等信息。"
                        },
                        "SwitchFontSize": {
                            "type": "Boolean",
                            "description": "用于识别用户想要调整字体大小的语音指令。该意图能够理解用户通过自然语言表达的字体大小调整需求，包括放大、缩小、设置特定大小等操作。"
                        },
                        "SwitchFontWeight": {
                            "type": "Boolean",
                            "description": "用于识别用户想要调整字体粗细的语音指令。该意图能够理解用户通过自然语言表达的字体粗细调整需求，包括设置粗体、细体、正常粗细等操作。"
                        },
                        "ControlFontType": {
                            "type": "Boolean",
                            "description": "用于识别用户想要控制字体类型或输入法字体的语音指令。该意图能够理解用户通过自然语言表达的字体类型切换需求，包括更换系统字体、应用字体、输入法字体等操作。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "performSearch",
                "description": "系统内搜索，查询内容",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Search": {
                            "type": "Boolean",
                            "description": "用于识别用户想要进行搜索操作的语音指令。该意图能够理解用户通过自然语言表达的搜索需求，包括网络搜索、本地搜索、应用内搜索、文件搜索等各种搜索相关的操作指令。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "setThemeAndWallpaper",
                "description": "设置主题/壁纸/模式",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "SetSystemTheme": {
                            "type": "Boolean",
                            "description": "用于识别用户想要设置系统主题的语音指令。该意图能够理解用户通过自然语言表达的主题切换需求，包括切换深色模式、浅色模式、自动模式，以及设置个性化主题等操作。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "manageMessages",
                "description": "发送短信或消息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "SendMessage": {
                            "type": "Boolean",
                            "description": "用于识别用户想要发送消息的语音指令。该意图能够理解用户通过自然语言表达的消息发送需求，包括在指定应用中发送消息、选择联系人、输入手机号码、选择手机卡以及编写消息内容等操作。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "manageBluetooth",
                "description": "开关蓝牙、搜索并连接设备",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "BlueToothOnOff": {
                            "type": "Boolean",
                            "description": "用于识别用户想要控制设备蓝牙功能开启或关闭的语音指令。该意图能够理解用户通过自然语言表达的蓝牙操作需求，包括但不限于打开蓝牙、关闭蓝牙、启用蓝牙连接、禁用蓝牙等相关操作指令。"
                        },
                        "SearchBlueTooth": {
                            "type": "Boolean",
                            "description": "用于识别用户想要搜索、查找或扫描附近可用蓝牙设备的语音指令。该意图能够理解用户通过自然语言表达的蓝牙设备发现需求，包括搜索蓝牙设备、扫描蓝牙、查找可连接设备等相关操作指令。"
                        },
                        "ConnectBlueTooth": {
                            "type": "Boolean",
                            "description": "用于识别用户想要连接或断开特定蓝牙设备的语音指令。该意图能够理解用户通过自然语言表达的蓝牙设备连接需求，包括连接蓝牙设备、断开蓝牙连接、配对设备等相关操作指令。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "manageSystemUpdates",
                "description": "检查并下载/安装系统更新",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "CheckSystemUpdate": {
                            "type": "Boolean",
                            "description": "用于识别用户想要检查系统更新的语音指令。该意图能够理解用户通过自然语言表达的系统更新查询需求，包括检查系统更新、查看更新状态、查询更新历史等操作。"
                        },
                        "SystemUpdate": {
                            "type": "Boolean",
                            "description": "用于识别用户想要执行系统更新的语音指令。该意图能够理解用户通过自然语言表达的系统更新执行需求，包括下载更新、安装更新、暂停更新、取消更新等操作。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "setLanguageAndInput",
                "description": "切换系统语言与输入法",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "CheckInputMethod": {
                            "type": "Boolean",
                            "description": "用于识别用户想要查看或检查输入法设置的语音指令。该意图能够理解用户通过自然语言表达的输入法查询需求，包括查看当前输入法、输入法设置、输入法列表、输入法状态等信息。"
                        },
                        "SwitchInputMethod": {
                            "type": "Boolean",
                            "description": "用于识别用户想要切换输入法的语音指令。该意图能够理解用户通过自然语言表达的输入法切换需求，包括切换到特定输入法、在应用中切换输入法、设置默认输入法等操作。"
                        },
                        "SwitchLanguage": {
                            "type": "Boolean",
                            "description": "用于识别用户想要切换语言设置的语音指令。该意图能够理解用户通过自然语言表达的语言切换需求，包括切换系统语言、应用语言、输入法语言等操作。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "setPowerSavingMode",
                "description": "开启或配置省电模式",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "BatterySavingMode": {
                            "type": "Boolean",
                            "description": "用于识别用户想要控制设备省电模式功能的语音指令。该意图能够理解用户通过自然语言表达的电池节能操作需求，包括开启省电模式、关闭节能模式、启用低电量模式、设置电池优化等相关操作指令。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "adjustDisplay",
                "description": "查询/调节亮度或对比度或自动亮度",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "CheckLuminance": {
                            "type": "Boolean",
                            "description": "用于识别用户想要查询设备当前屏幕亮度的语音指令。该意图能够理解用户通过自然语言表达的亮度查询需求，包括查看当前亮度、检查屏幕亮度、显示亮度等级、查询亮度设置等相关查询指令。"
                        },
                        "ControlLuminance": {
                            "type": "Boolean",
                            "description": "用于识别用户想要控制设备屏幕亮度的语音指令。该意图能够理解用户通过自然语言表达的亮度调节需求，包括调节屏幕亮度、增加/减少亮度、设置亮度百分比、调整显示器亮度等相关操作指令。"
                        },
                        "AutoLuminanceOnOff": {
                            "type": "Boolean",
                            "description": "用于识别用户想要控制设备自动亮度调节功能开启或关闭的语音指令。该意图能够理解用户通过自然语言表达的自动亮度功能操作需求，包括开启自动亮度、关闭自适应亮度、启用智能亮度调节、禁用自动亮度等相关操作指令。"
                        },
                        "CheckContrast": {
                            "type": "Boolean",
                            "description": "用于识别用户想要查询设备当前屏幕对比度的语音指令。该意图能够理解用户通过自然语言表达的对比度查询需求，包括查看当前对比度、检查屏幕对比度、显示对比度等级、查询对比度设置等相关查询指令。"
                        },
                        "SwitchContrast": {
                            "type": "Boolean",
                            "description": "用于识别用户想要切换设备屏幕对比度模式的语音指令。该意图能够理解用户通过自然语言表达的对比度模式切换需求，包括切换到高对比度、标准对比度、低对比度、护眼模式等不同对比度模式的操作指令。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "getBatteryStatus",
                "description": "查询电量与耗电排行",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "CheckBatteryLevel": {
                            "type": "Boolean",
                            "description": "用于识别用户想要查询设备电池电量状态的语音指令。该意图能够理解用户通过自然语言表达的电池信息查询需求，包括查看电池电量、询问剩余电量、检查电池状态、了解续航时间等相关查询指令。"
                        },
                        "CheckBatteryConsumptionRank": {
                            "type": "Boolean",
                            "description": "用于识别用户想要查询设备电池消耗排名和耗电分析的语音指令。该意图能够理解用户通过自然语言表达的电池使用情况分析需求，包括查看应用耗电排行、分析电池消耗趋势、检测异常耗电、对比不同时间段的电池使用情况等相关查询指令。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "refreshUI",
                "description": "界面刷新/刷新界面视图",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Refresh": {
                            "type": "Boolean",
                            "description": "用于识别用户想要执行刷新操作的语音指令。该意图能够理解用户通过自然语言表达的刷新需求，包括刷新页面、更新内容、重新加载、刷新桌面、刷新文件夹等各种刷新相关的操作指令。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "toggleAirplaneMode",
                "description": "开启/关闭/切换飞行模式",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "AirplaneModeOnOff": {
                            "type": "Boolean",
                            "description": "用于识别用户想要控制飞行模式开关的语音指令。该意图能够理解用户通过自然语言表达的飞行模式控制需求，包括开启、关闭、切换飞行模式等操作。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "controlNetwork",
                "description": "开关WLAN/移动数据、搜索Wi-Fi、热点共享",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "WlanOnOff": {
                            "type": "Boolean",
                            "description": "用于识别用户想要控制设备WLAN（无线局域网）功能开启或关闭的语音指令。该意图能够理解用户通过自然语言表达的WiFi操作需求，包括打开WiFi、关闭WiFi、启用无线网络、禁用WLAN等相关操作指令。"
                        },
                        "SearchWlan": {
                            "type": "Boolean",
                            "description": "用于识别用户想要搜索、查找或扫描附近可用WLAN（无线局域网）网络的语音指令。该意图能够理解用户通过自然语言表达的WiFi网络发现需求，包括搜索WiFi、扫描无线网络、查找可连接热点等相关操作指令。"
                        },
                        "MobileDataOnOff": {
                            "type": "Boolean",
                            "description": "用于识别用户想要控制设备移动数据网络功能开启或关闭的语音指令。该意图能够理解用户通过自然语言表达的移动网络操作需求，包括打开移动数据、关闭蜂窝数据、启用4G/5G网络、禁用流量等相关操作指令。"
                        },
                        "HotShotOnOff": {
                            "type": "Boolean",
                            "description": "用于识别用户想要控制设备热点（WiFi热点/个人热点）功能开启或关闭的语音指令。该意图能够理解用户通过自然语言表达的热点分享操作需求，包括打开热点、关闭热点、启用WiFi共享、禁用个人热点等相关操作指令。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "manageEmail",
                "description": "发送或查看邮件",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "SendEmail": {
                            "type": "Boolean",
                            "description": "用于识别用户想要发送邮件的语音指令。该意图能够理解用户通过自然语言表达的邮件发送需求，包括在指定邮件应用中发送邮件、选择收件人联系人、输入邮箱地址以及编写邮件内容等操作。"
                        },
                        "CheckEmail": {
                            "type": "Boolean",
                            "description": "用于识别用户想要查看或检查邮件的语音指令。该意图能够理解用户通过自然语言表达的邮件查询需求，包括在指定邮件应用中查看邮件、按文件夹分类查看、按发件人筛选、按内容搜索以及按时间范围查询等操作。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "manageRecycleBin",
                "description": "清空回收站/废纸篓",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "EmptyBin": {
                            "type": "Boolean",
                            "description": "用于识别用户想要清空回收站或垃圾箱的语音指令。该意图能够理解用户通过自然语言表达的清空垃圾箱需求，包括清空回收站、删除垃圾文件、清理废纸篓、永久删除回收站文件等相关操作指令。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "switchApp",
                "description": "打开某应用或软件/在前台应用或软件之间切换",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "SwitchApp": {
                            "type": "Boolean",
                            "description": "用于识别用户想要切换或打开应用程序的语音指令。该意图能够理解用户通过自然语言表达的应用切换需求，包括打开特定应用、在应用间切换等操作。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "launchApp",
                "description": "翻译文本或截屏保存",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Translate": {
                            "type": "Boolean",
                            "description": "用于识别用户想要进行翻译操作的语音指令。该意图能够理解用户通过自然语言表达的翻译需求，包括文本翻译、语言转换、多语言互译等各种翻译相关的操作指令。"
                        },
                        "CaptureScreenshot": {
                            "type": "Boolean",
                            "description": "用于识别用户想要进行静态屏幕截图的语音指令。该意图能够理解用户通过自然语言表达的截图需求，包括全屏截图、区域截图、窗口截图等各种屏幕捕获操作，并支持指定截图区域和保存路径。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "manageAlarms",
                "description": "新建/查看/删除闹钟",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "CreateAlarm": {
                            "type": "Boolean",
                            "description": "用于识别用户想要创建或设置闹钟的语音指令。该意图能够理解用户通过自然语言表达的闹钟创建需求，包括设置特定时间的闹钟、为特定事件创建提醒、设置重复闹钟以及配置闹钟的各种属性等操作。"
                        },
                        "CheckAlarm": {
                            "type": "Boolean",
                            "description": "用于识别用户想要查看或检查闹钟状态的语音指令。该意图能够理解用户通过自然语言表达的查询闹钟需求，包括查看所有闹钟、检查特定时间的闹钟、查询特定事件的提醒、查看闹钟状态等操作。"
                        },
                        "DeleteAlarm": {
                            "type": "Boolean",
                            "description": "用于识别用户想要删除闹钟或提醒的语音指令。该意图能够理解用户通过自然语言表达的删除闹钟需求，包括删除特定时间的闹钟、删除特定事件的提醒、批量删除闹钟以及删除特定状态的闹钟等操作。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "showTaskSwitcher",
                "description": "打开任务管理器或切换多窗口",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "TaskManagerOnOff": {
                            "type": "Boolean",
                            "description": "用于识别用户想要控制任务管理器开启或关闭的语音指令。该意图能够理解用户通过自然语言表达的任务管理器操作需求，包括打开任务管理器、关闭任务管理器、启动进程管理器、退出系统监视器等相关操作指令。"
                        },
                        "MultipleWindowModeOnOff": {
                            "type": "Boolean",
                            "description": "用于识别用户想要控制多窗口模式开启或关闭的语音指令。该意图能够理解用户通过自然语言表达的多窗口功能操作需求，包括开启分屏模式、关闭多窗口显示、启用窗口并排显示、禁用分割屏幕等相关操作指令。"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "toggleSoundMode",
                "description": "切换振动/勿扰/静音模式",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "VibrationModeOnOff": {
                            "type": "Boolean",
                            "description": "用于识别用户想要控制设备振动模式开启或关闭的语音指令。该意图能够理解用户通过自然语言表达的振动功能操作需求，包括开启振动、关闭震动、启用震动模式、禁用振动提醒等相关操作指令。"
                        },
                        "QuietModeOnOff": {
                            "type": "Boolean",
                            "description": "用于识别用户想要控制设备勿扰模式（静音模式）开启或关闭的语音指令。该意图能够理解用户通过自然语言表达的勿扰功能操作需求，包括开启勿扰模式、关闭静音模式、启用免打扰、禁用勿扰等相关操作指令。"
                        }
                    },
                    "required": []
                }
            }
        }

]

# 从工具定义中提取分类信息
level1_tools = {}
for tool in tools_definition:
    function = tool["function"]
    tool_name = function["name"]
    description = function["description"]
    parameters = list(function["parameters"]["properties"].keys())

    level1_tools[tool_name] = {
        'description': description,
        'parameters': parameters
    }

# 输出文件路径
output_file = "complex_level1_classification_lora_dataset.json"
batch_size = 8
total_batches = 2
all_samples = []


# -------------------------- 2. 生成复杂场景的分类数据集 --------------------------
def generate_complex_level1_classification_batch(batch_num, batch_size, level1_tools):
    """生成复杂真实场景的第一级分类训练数据"""

    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(base_url=BASE_URL)
    )

    # 构建详细的工具类别说明
    category_details = []
    for tool_name, info in level1_tools.items():
        params_text = "、".join(info['parameters'])
        category_details.append(f"- {tool_name}: {info['description']}，可用参数: {params_text}")

    category_text = "\n".join(category_details)

    prompt = f"""
你是一个工具分类数据生成器。需要为第一级工具分类模型生成训练数据，重点是生成**复杂、真实、自然的用户指令**。

## 任务：
生成{batch_size}条**复杂真实场景**的用户指令与对应工具分类的样本数据。

## 工具类别系统（基于实际定义）：
{category_text}

## 输出格式要求：
返回一个JSON数组，每个元素包含：
{{
    "instruction": "复杂真实的用户指令", 
    "input": "",
    "output": "<level>工具类别名(参数名=True)</level>"
}}

## 生成复杂指令的要求：
1. **场景复杂性**：包含具体情境、原因、背景信息
2. **语言自然性**：使用口语化、自然的表达方式
3. **多样性**：涵盖不同场景、不同表达方式
4. **真实感**：模拟真实用户会说的话

## 复杂指令示例（参考这些风格）：
- "我手机快没流量了，帮我把移动网络关掉吧" → controlNetwork
- "帮我看看周围有什么WiFi能用" → controlNetwork  
- "帮我把平板屏幕的亮度调到60%吧，有点刺眼" → adjustDisplay
- "手机电量只剩15%了，赶紧开个省电模式" → setPowerSavingMode
- "这个蓝牙耳机老是断连，重新搜索配对吧" → manageBluetooth
- "明天早上要赶飞机，设个6点的闹钟提醒我" → manageAlarms
- "屏幕上的字太小了，老人家看不清楚，调大点" → setFont
- "手机用着有点卡，清理下缓存吧" → runSystemUtility

## 参数选择指南：
- **powerDevice**：
  * "手机死机了，强制重启一下" → Reboot=True
  * "要上飞机了，先把设备关机" → ShutDown=True  
  * "暂时不用手机，让它锁屏休眠吧" → Sleep=True

- **placeCall**：
  * "帮我打给客户经理张先生" → Call=True
  * "找一下李老师的电话号码" → CheckContact=True
  * "有来电显示，接听一下" → AnswerCall=True

- **manageBluetooth**：
  * "连接不上车载蓝牙，重新开关试试" → BlueToothOnOff=True
  * "搜索下附近有没有我的蓝牙音箱" → SearchBlueTooth=True
  * "这个新耳机要配对连接" → ConnectBlueTooth=True

- **controlNetwork**：
  * "流量超了，关掉移动数据" → MobileDataOnOff=True
  * "看看酒店有什么WiFi可以连" → SearchWlan=True
  * "开个热点给笔记本用" → HotShotOnOff=True

- **adjustDisplay**：
  * "太阳底下屏幕反光，调亮一点" → ControlLuminance=True
  * "晚上看手机太亮，开自动亮度" → AutoLuminanceOnOff=True
  * "这个显示模式看着不舒服，换个对比度" → SwitchContrast=True

## 更多复杂场景思路：
- 包含具体原因："因为...所以..."
- 包含时间信息："明天早上...","现在..."
- 包含设备状态："手机快没电了","网络信号不好"
- 包含用户感受："太刺眼了","看不清楚","用着卡顿"
- 包含具体对象："给奶奶的手机","平板的屏幕"

## 示例输出：
[
    {{
        "instruction": "手机存储空间不足了，帮我清理下系统缓存腾点地方",
        "input": "",
        "output": "<level>runSystemUtility(CreateNote=True)</level>"
    }},
    {{
        "instruction": "晚上关灯玩手机太伤眼睛，开个深色模式保护下视力",
        "input": "",
        "output": "<level>setThemeAndWallpaper(SetSystemTheme=True)</level>"
    }},
    {{
        "instruction": "这个英文界面看不懂，切换成中文显示吧",
        "input": "",
        "output": "<level>setLanguageAndInput(SwitchLanguage=True)</level>"
    }},
    {{
        "instruction": "开会时手机老是响，调成振动模式别打扰别人",
        "input": "",
        "output": "<level>toggleSoundMode(VibrationModeOnOff=True)</level>"
    }},
    {{
        "instruction": "给孩子设个学习时间的闹钟，每天下午4点提醒",
        "input": "",
        "output": "<level>manageAlarms(CreateAlarm=True)</level>"
    }},
    {{
        "instruction": "查一下哪个应用最耗电，优化下电池使用",
        "input": "",
        "output": "<level>getBatteryStatus(CheckBatteryConsumptionRank=True)</level>"
    }},
    {{
        "instruction": "网页内容没更新，刷新一下看看最新消息",
        "input": "",
        "output": "<level>refreshUI(Refresh=True)</level>"
    }},
    {{
        "instruction": "要给客户发个重要邮件，帮我打开邮箱应用",
        "input": "",
        "output": "<level>manageEmail(SendEmail=True)</level>"
    }}
]

请直接返回JSON数组，不要有其他文字说明。重点生成复杂真实的用户指令！
"""

    try:
        response = client.models.generate_content(
            model='models/gemini-2.5-pro',
            contents=types.Content(
                parts=[
                    types.Part(text=prompt)
                ]
            )
        )

        response_text = response.text.strip()
        print(f"第{batch_num}批原始响应: {response_text[:200]}...")

        try:
            # 清理响应文本
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            batch_samples = json.loads(response_text)

            # 验证逻辑
            valid_samples = []
            for sample in batch_samples:
                if (isinstance(sample, dict) and
                        "instruction" in sample and
                        "input" in sample and
                        "output" in sample):

                    output = sample["output"]

                    # 验证格式
                    if (output.startswith("<level>") and
                            output.endswith("</level>") and
                            "(" in output and ")" in output and
                            "=True" in output):

                        # 提取类别和参数
                        try:
                            content = output[7:-8]  # 去掉<level>和</level>
                            category_end = content.find("(")
                            category = content[:category_end]
                            param_content = content[category_end + 1:-1]  # 去掉括号
                            param = param_content.split("=")[0]

                            # 验证类别和参数是否存在
                            if (category in level1_tools and
                                    param in level1_tools[category]['parameters']):
                                valid_samples.append(sample)
                                print(f"✓ 有效样本: {sample['instruction'][:30]}... → {category}.{param}")
                            else:
                                print(f"✗ 无效的类别或参数：{category}.{param}，跳过样本")
                                continue

                        except Exception as e:
                            print(f"✗ 解析输出时出错：{output}，错误：{e}")
                            continue
                    else:
                        print(f"✗ 格式不正确：{output}")

                else:
                    print(f"✗ 样本格式不完整：{sample}")

            return valid_samples

        except json.JSONDecodeError as e:
            print(f"JSON解析失败（第{batch_num}批）：{e}")
            print(f"原始响应：{response_text}")
            return []

    except Exception as e:
        print(f"API调用失败（第{batch_num}批）：{str(e)}")
        time.sleep(5)
        return []


# -------------------------- 3. 主函数 --------------------------
def main():
    """生成复杂真实场景的第一级分类训练数据集"""
    print("开始生成复杂真实场景的第一级分类训练数据集...")
    print(f"共 {len(level1_tools)} 个工具类别")

    for batch in range(1, total_batches + 1):
        print(f"\n正在生成第{batch}批样本...")
        batch_samples = generate_complex_level1_classification_batch(batch, batch_size, level1_tools)
        if batch_samples:
            all_samples.extend(batch_samples)
            print(f"第{batch}批生成成功，获得{len(batch_samples)}条样本，累计{len(all_samples)}条")
        else:
            print(f"第{batch}批生成失败")
        time.sleep(2)

    # 保存数据
    if all_samples:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)

        print(f"\n数据集生成完成！共 {len(all_samples)} 条复杂样本")

        # 统计分类分布
        category_stats = {}
        param_stats = {}

        for sample in all_samples:
            output = sample["output"]
            try:
                content = output[7:-8]  # 去掉<level>和</level>
                category_end = content.find("(")
                category = content[:category_end]
                param_content = content[category_end + 1:-1]
                param = param_content.split("=")[0]

                category_stats[category] = category_stats.get(category, 0) + 1
                param_key = f"{category}.{param}"
                param_stats[param_key] = param_stats.get(param_key, 0) + 1
            except:
                continue

        print("\n工具类别分布：")
        for category, count in sorted(category_stats.items()):
            print(f"  {category}: {count}条")

        print("\n参数使用统计（前15个）：")
        for param, count in sorted(param_stats.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {param}: {count}条")

        print(f"\n总样本数: {len(all_samples)}")
        print(f"已保存到: {output_file}")

        # 显示一些复杂样本示例
        print("\n复杂样本示例：")
        for i, sample in enumerate(all_samples[:5]):
            print(f"  {i + 1}. {sample['instruction']}")
            print(f"     → {sample['output']}")

    else:
        print("未生成任何有效样本")


if __name__ == "__main__":
    main()