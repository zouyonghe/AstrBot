from flask import Flask, request
from flask.logging import default_handler
from werkzeug.serving import make_server
import datetime
from util import general_utils as gu
from dataclasses import dataclass
import logging

@dataclass
class DashBoardData():
    stats: dict
    configs: dict
    logs: dict

@dataclass
class Response():
    status: str
    message: str
    data: dict
    
class AstrBotDashBoard():
    def __init__(self, dashboard_data: DashBoardData):
        self.dashboard_data = dashboard_data
        self.dashboard_be = Flask(__name__, static_folder="dist", static_url_path="/")
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        self.funcs = {}
        
        
        @self.dashboard_be.get("/")
        def index():
            # 返回页面
            return self.dashboard_be.send_static_file("index.html")
        
        # 如果是以.js结尾的
        # @self.dashboard_be.get("/<path:path>.js")
        # def js(path):
        #     return self.dashboard_be.send_static_file(path + ".js")
        
        

        @self.dashboard_be.get("/api/stats")
        def get_stats():
            return Response(
                status="success",
                message="",
                data=self.dashboard_data.stats
            ).__dict__
        
        @self.dashboard_be.get("/api/configs")
        def get_configs():
            return Response(
                status="success",
                message="",
                data=self.dashboard_data.configs
            ).__dict__
            
        @self.dashboard_be.post("/api/configs")
        def post_configs():
            post_configs = request.json
            try:
                self.funcs["post_configs"](post_configs)
                return Response(
                    status="success",
                    message="保存成功~ 机器人将在 2 秒内重启以应用新的配置。",
                    data=None
                ).__dict__
            except Exception as e:
                return Response(
                    status="error",
                    message=e.__str__(),
                    data=self.dashboard_data.configs
                ).__dict__
        
        @self.dashboard_be.get("/api/logs")
        def get_logs():
            return Response(
                status="success",
                message="",
                data=self.dashboard_data.logs
            ).__dict__
        
    def register(self, name: str):
        def decorator(func):
            self.funcs[name] = func
            return func
        return decorator

    def run(self):
        gu.log(f"\n\n==================\n您可以访问:\n\thttp://localhost:6185/\n来登录可视化面板。\n==================\n\n", tag="可视化面板")
        # self.dashboard_be.run(host="0.0.0.0", port=6185)
        http_server = make_server('0.0.0.0', 6185, self.dashboard_be)
        http_server.serve_forever()
