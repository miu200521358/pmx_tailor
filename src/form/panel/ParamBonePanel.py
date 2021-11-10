# -*- coding: utf-8 -*-
#
import wx

from form.panel.BasePanel import BasePanel
from utils.MLogger import MLogger # noqa

logger = MLogger(__name__)


class ParamBonePanel(BasePanel):
        
    def __init__(self, frame: wx.Frame, export: wx.Notebook, tab_idx: int):
        super().__init__(frame, export, tab_idx)
        self.convert_export_worker = None

        self.header_panel = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        self.header_sizer = wx.BoxSizer(wx.VERTICAL)

        self.description_txt = wx.StaticText(self, wx.ID_ANY, u"パラ調整タブで選択された材質に既にボーンとウェイトが設定されている場合に、\n" \
                                             u"ボーン構成を指定する事で物理設定（剛体・ジョイント）を設定する事が出来ます。\n" \
                                             u"縦方向がボーンの親子関係、横がボーンの並び順で指定してください。\n" \
                                             u"（スカートなどは水平方向の並び順、袖などは輪切り縦方向の並び順を横方向に並べてください）\n" \
                                             u"ボーン名を指定すると、その子ボーンを自動設定します。（少しタイムラグがあります）\n" \
                                             u"水平方向のボーン間が繋がっているかは、ウェイトを見て自動判定します。", wx.DefaultPosition, wx.DefaultSize, 0)
        self.header_sizer.Add(self.description_txt, 0, wx.ALL, 5)

        self.static_line01 = wx.StaticLine(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL)
        self.header_sizer.Add(self.static_line01, 0, wx.EXPAND | wx.ALL, 5)

        self.header_panel.SetSizer(self.header_sizer)
        self.header_panel.Layout()
        self.sizer.Add(self.header_panel, 0, wx.EXPAND | wx.ALL, 5)

        # 詳細Sizer
        self.bone_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.scrolled_window = wx.ScrolledWindow(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.FULL_REPAINT_ON_RESIZE | wx.VSCROLL | wx.HSCROLL | wx.ALWAYS_SHOW_SB)
        self.scrolled_window.SetScrollRate(5, 5)

        self.scrolled_window.SetSizer(self.bone_sizer)
        self.scrolled_window.Layout()
        self.sizer.Add(self.scrolled_window, 1, wx.ALL | wx.EXPAND | wx.FIXED_MINSIZE, 5)
        self.fit()

        self.Layout()
        self.fit()

    def initialize(self, event: wx.Event):
        self.frame.simple_param_panel_ctrl.initialize(event)

