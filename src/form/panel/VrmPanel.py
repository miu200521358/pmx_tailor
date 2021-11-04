# -*- coding: utf-8 -*-
#
import wx
import wx.lib.newevent

from form.panel.BasePanel import BasePanel
from form.parts.ConsoleCtrl import ConsoleCtrl
from utils import MFormUtils, MFileUtils
from utils.MLogger import MLogger # noqa

logger = MLogger(__name__)
TIMER_ID = wx.NewId()

# イベント定義
(ParentThreadEvent, EVT_SMOOTH_THREAD) = wx.lib.newevent.NewEvent()


class VrmPanel(BasePanel):
        
    def __init__(self, frame: wx.Frame, export: wx.Notebook, tab_idx: int):
        super().__init__(frame, export, tab_idx)
        self.convert_export_worker = None

        self.header_panel = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        self.header_sizer = wx.BoxSizer(wx.VERTICAL)

        self.description_txt = wx.StaticText(self, wx.ID_ANY, u"Vroid Studio 正式版(v1.0.0以降)で作成されたVrmに関する設定を行ってください", wx.DefaultPosition, wx.DefaultSize, 0)
        self.header_sizer.Add(self.description_txt, 0, wx.ALL, 5)

        self.static_line01 = wx.StaticLine(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL)
        self.header_sizer.Add(self.static_line01, 0, wx.EXPAND | wx.ALL, 5)

        self.header_panel.SetSizer(self.header_sizer)
        self.header_panel.Layout()
        self.sizer.Add(self.header_panel, 0, wx.EXPAND | wx.ALL, 5)

        # 詳細Sizer
        self.advance_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.scrolled_window = wx.ScrolledWindow(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.FULL_REPAINT_ON_RESIZE | wx.VSCROLL | wx.ALWAYS_SHOW_SB)
        self.scrolled_window.SetScrollRate(5, 5)

        self.scrolled_window.SetSizer(self.advance_sizer)
        self.scrolled_window.Layout()
        self.sizer.Add(self.scrolled_window, 1, wx.ALL | wx.EXPAND | wx.FIXED_MINSIZE, 5)
        self.fit()

        self.Layout()
        self.fit()

    def initialize(self, event: wx.Event):
        self.frame.simple_param_panel_ctrl.initialize(event)


BONE_PAIRS = {
    "Root": {"name": "全ての親", "parent": None, "tail": "センター", "display": "全ての親", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "Center": {"name": "センター", "parent": "全ての親", "tail": -1, "display": "センター", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "Groove": {"name": "グルーブ", "parent": "センター", "tail": -1, "display": "センター", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_C_Hips": {"name": "腰", "parent": "グルーブ", "tail": -1, "display": "体幹", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": 0, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Bip_C_Spine": {"name": "下半身", "parent": "腰", "tail": -1, "display": "体幹", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 0, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Bip_C_Chest": {"name": "上半身", "parent": "腰", "tail": "上半身2", "display": "体幹", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 0, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Bip_C_UpperChest": {"name": "上半身2", "parent": "上半身", "tail": "首", "display": "体幹", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 0, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Bip_C_Neck": {"name": "首", "parent": "上半身2", "tail": "頭", "display": "体幹", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 0, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Bip_C_Head": {"name": "頭", "parent": "首", "tail": -1, "display": "体幹", "rigidBody": 0, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 0, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Adj_FaceEye": {"name": "両目", "parent": "頭", "tail": -1, "display": "顔", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Adj_L_FaceEye": {"name": "左目", "parent": "頭", "tail": -1, "display": "顔", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Adj_R_FaceEye": {"name": "右目", "parent": "頭", "tail": -1, "display": "顔", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Sec_L_Bust1": {"name": "左胸", "parent": "上半身2", "tail": "左胸先", "display": "胸", "rigidBody": 0, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 0, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Sec_L_Bust2": {"name": "左胸先", "parent": "左胸", "tail": -1, "display": None, "rigidBody": 0, "rigidbodyMode": 2, "rigidbodyFactor": 1, "rigidbodyCollisions": 0, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Sec_R_Bust1": {"name": "右胸", "parent": "上半身2", "tail": "右胸先", "display": "胸", "rigidBody": 0, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 0, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Sec_R_Bust2": {"name": "右胸先", "parent": "右胸", "tail": -1, "display": None, "rigidBody": 0, "rigidbodyMode": 2, "rigidbodyFactor": 1, "rigidbodyCollisions": 0, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "shoulderP_L": {"name": "左肩P", "parent": "上半身2", "tail": -1, "display": "左手", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Shoulder": {"name": "左肩", "parent": "左肩P", "tail": "左腕", "display": "左手", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 1, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "shoulderC_L": {"name": "左肩C", "parent": "左肩", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_UpperArm": {"name": "左腕", "parent": "左肩C", "tail": "左ひじ", "display": "左手", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 1, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "arm_twist_L": {"name": "左腕捩", "parent": "左腕", "tail": -1, "display": "左手", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "arm_twist_L1": {"name": "左腕捩1", "parent": "左腕", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "arm_twist_L2": {"name": "左腕捩2", "parent": "左腕", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "arm_twist_L3": {"name": "左腕捩3", "parent": "左腕", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_LowerArm": {"name": "左ひじ", "parent": "左腕捩", "tail": "左手首", "display": "左手", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 1, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "wrist_twist_L": {"name": "左手捩", "parent": "左ひじ", "tail": -1, "display": "左手", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "wrist_twist_L1": {"name": "左手捩1", "parent": "左ひじ", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "wrist_twist_L2": {"name": "左手捩2", "parent": "左ひじ", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "wrist_twist_L3": {"name": "左手捩3", "parent": "左ひじ", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Hand": {"name": "左手首", "parent": "左手捩", "tail": -1, "display": "左手", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 1, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Bip_L_Thumb1": {"name": "左親指０", "parent": "左手首", "tail": "左親指１", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Thumb2": {"name": "左親指１", "parent": "左親指０", "tail": "左親指２", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Thumb3": {"name": "左親指２", "parent": "左親指１", "tail": "左親指先", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Thumb3_end": {"name": "左親指先", "parent": "左親指２", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Index1": {"name": "左人指１", "parent": "左手首", "tail": "左人指２", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Index2": {"name": "左人指２", "parent": "左人指１", "tail": "左人指３", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Index3": {"name": "左人指３", "parent": "左人指２", "tail": "左人指先", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Index3_end": {"name": "左人指先", "parent": "左人指３", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Middle1": {"name": "左中指１", "parent": "左手首", "tail": "左中指２", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Middle2": {"name": "左中指２", "parent": "左中指１", "tail": "左中指３", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Middle3": {"name": "左中指３", "parent": "左中指２", "tail": "左中指先", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Middle3_end": {"name": "左中指先", "parent": "左中指３", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Ring1": {"name": "左薬指１", "parent": "左手首", "tail": "左薬指２", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Ring2": {"name": "左薬指２", "parent": "左薬指１", "tail": "左薬指３", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Ring3": {"name": "左薬指３", "parent": "左薬指２", "tail": "左薬指先", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Ring3_end": {"name": "左薬指先", "parent": "左薬指３", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Little1": {"name": "左小指１", "parent": "左手首", "tail": "左小指２", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Little2": {"name": "左小指２", "parent": "左小指１", "tail": "左小指３", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Little3": {"name": "左小指３", "parent": "左小指２", "tail": "左小指先", "display": "左指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_Little3_end": {"name": "左小指先", "parent": "左小指３", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "shoulderP_R": {"name": "右肩P", "parent": "上半身2", "tail": -1, "display": "右手", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Shoulder": {"name": "右肩", "parent": "右肩P", "tail": "右腕", "display": "右手", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 1, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "shoulderC_R": {"name": "右肩C", "parent": "右肩", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_UpperArm": {"name": "右腕", "parent": "右肩C", "tail": "右ひじ", "display": "右手", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 1, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "arm_twist_R": {"name": "右腕捩", "parent": "右腕", "tail": -1, "display": "右手", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "arm_twist_R1": {"name": "右腕捩1", "parent": "右腕", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "arm_twist_R2": {"name": "右腕捩2", "parent": "右腕", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "arm_twist_R3": {"name": "右腕捩3", "parent": "右腕", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_LowerArm": {"name": "右ひじ", "parent": "右腕捩", "tail": "右手首", "display": "右手", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 1, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "wrist_twist_R": {"name": "右手捩", "parent": "右ひじ", "tail": -1, "display": "右手", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "wrist_twist_R1": {"name": "右手捩1", "parent": "右ひじ", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "wrist_twist_R2": {"name": "右手捩2", "parent": "右ひじ", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "wrist_twist_R3": {"name": "右手捩3", "parent": "右ひじ", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Hand": {"name": "右手首", "parent": "右手捩", "tail": -1, "display": "右手", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 1, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Bip_R_Thumb1": {"name": "右親指０", "parent": "右手首", "tail": "右親指１", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Thumb2": {"name": "右親指１", "parent": "右親指０", "tail": "右親指２", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Thumb3": {"name": "右親指２", "parent": "右親指１", "tail": "右親指先", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Thumb3_end": {"name": "右親指先", "parent": "右親指２", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Index1": {"name": "右人指１", "parent": "右手首", "tail": "右人指２", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Index2": {"name": "右人指２", "parent": "右人指１", "tail": "右人指３", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Index3": {"name": "右人指３", "parent": "右人指２", "tail": "右人指先", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Index3_end": {"name": "右人指先", "parent": "右人指３", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Middle1": {"name": "右中指１", "parent": "右手首", "tail": "右中指２", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Middle2": {"name": "右中指２", "parent": "右中指１", "tail": "右中指３", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Middle3": {"name": "右中指３", "parent": "右中指２", "tail": "右中指先", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Middle3_end": {"name": "右中指先", "parent": "右中指３", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Ring1": {"name": "右薬指１", "parent": "右手首", "tail": "右薬指２", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Ring2": {"name": "右薬指２", "parent": "右薬指１", "tail": "右薬指３", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Ring3": {"name": "右薬指３", "parent": "右薬指２", "tail": "右薬指先", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Ring3_end": {"name": "右薬指先", "parent": "右薬指３", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Little1": {"name": "右小指１", "parent": "右手首", "tail": "右小指２", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Little2": {"name": "右小指２", "parent": "右小指１", "tail": "右小指３", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Little3": {"name": "右小指３", "parent": "右小指２", "tail": "右小指先", "display": "右指", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_Little3_end": {"name": "右小指先", "parent": "右小指３", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "leftWaistCancel": {"name": "腰キャンセル左", "parent": "下半身", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_UpperLeg": {"name": "左足", "parent": "腰キャンセル左", "tail": "左ひざ", "display": "左足", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 2, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Bip_L_LowerLeg": {"name": "左ひざ", "parent": "左足", "tail": "左足首", "display": "左足", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 2, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Bip_L_Foot": {"name": "左足首", "parent": "左ひざ", "tail": "左つま先", "display": "左足", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 2, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Bip_L_ToeBase_end": {"name": "左つま先", "parent": "左足首", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "leg_IK_L": {"name": "左足ＩＫ", "parent": "全ての親", "tail": -1, "display": "左足", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "toe_IK_L": {"name": "左つま先ＩＫ", "parent": "左足ＩＫ", "tail": -1, "display": "左足", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "rightWaistCancel": {"name": "腰キャンセル右", "parent": "下半身", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_UpperLeg": {"name": "右足", "parent": "腰キャンセル右", "tail": "右ひざ", "display": "右足", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 2, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Bip_R_LowerLeg": {"name": "右ひざ", "parent": "右足", "tail": "右足首", "display": "右足", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 2, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Bip_R_Foot": {"name": "右足首", "parent": "右ひざ", "tail": "右つま先", "display": "右足", "rigidBody": 2, "rigidbodyMode": 0, "rigidbodyFactor": 1, "rigidbodyCollisions": 2, "rigidbodyNoCollisions": [0, 1, 2], "rigidbodyParam": [1, 0.5, 0.5, 0, 0.5]},
    "J_Bip_R_ToeBase_end": {"name": "右つま先", "parent": "右足首", "tail": -1, "display": None, "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "leg_IK_R": {"name": "右足ＩＫ", "parent": "全ての親", "tail": -1, "display": "右足", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "toe_IK_R": {"name": "右つま先ＩＫ", "parent": "右足ＩＫ", "tail": -1, "display": "右足", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "leg_LD": {"name": "左足D", "parent": "腰キャンセル左", "tail": -1, "display": "左足", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "knee_LD": {"name": "左ひざD", "parent": "左足D", "tail": -1, "display": "左足", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "ankle_LD": {"name": "左足首D", "parent": "左ひざD", "tail": -1, "display": "左足", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_L_ToeBase": {"name": "左足先EX", "parent": "左足首D", "tail": -1, "display": "左足", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "leg_RD": {"name": "右足D", "parent": "腰キャンセル右", "tail": -1, "display": "右足", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "knee_RD": {"name": "右ひざD", "parent": "右足D", "tail": -1, "display": "右足", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "ankle_RD": {"name": "右足首D", "parent": "右ひざD", "tail": -1, "display": "右足", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": [], "rigidbodyParam": []},
    "J_Bip_R_ToeBase": {"name": "右足先EX", "parent": "右足首D", "tail": -1, "display": "右足", "rigidBody": None, "rigidbodyMode": 0, "rigidbodyFactor": None, "rigidbodyCollisions": -1, "rigidbodyNoCollisions": []}
}

