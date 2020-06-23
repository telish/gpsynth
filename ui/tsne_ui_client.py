import argparse
import json
import os

from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *

from pythonosc import udp_client



class GP_Control(QWebEngineView):

    def __init__(self, score, should_play):
        super().__init__()

        self.score = score
        self.should_play = should_play

        # setup a page with my html
        my_page = QWebEnginePage(self)
        my_page.setUrl(QUrl("http://127.0.0.1:4555/fixed"))
        self.setPage(my_page)

        # setup channel
        self.channel = QWebChannel()
        self.channel.registerObject('backend', self)
        self.page().setWebChannel(self.channel)
        self.show()

    @pyqtSlot(float, result=float)
    def clicked(self, time):
        event = self.event_at(time)
        print('clicked', event)
        waveshaping = 'waveshaping_' if event['waveshaping'] else ''
        if event['operator'] == '':
            value = f"{waveshaping}{event['kernel_1']}_l{event['lengthscale_1_idx']:03d}_n{event['note']:02d}.wav"
        else:
            value = f"{waveshaping}{event['kernel_1']}_l{event['lengthscale_1_idx']:03d}({event['operator']})" \
                    f"{event['kernel_2']}_l{event['lengthscale_2_idx']:03d}_n{event['note']:02d}.wav"

        send_event_osc(value)
        return 42

    @pyqtSlot(result=bool)
    def should_play(self):
        return self.should_play

    def event_at(self, time):
        eps = 10e-6
        for event in self.score:
            if abs(event['time'] - time) < eps:
                return event


def send_event_osc(event):
    client = udp_client.SimpleUDPClient(send_event_osc.ip, send_event_osc.port)
    client.send_message("/loadtable", event)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument("--ip", default="127.0.0.1",
                        help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=5005,
                        help="The port the OSC server is listening on")
    parser.add_argument('--quiet', dest='quiet', action='store_true')
    args = parser.parse_args()

    send_event_osc.ip = args.ip
    send_event_osc.port = args.port

    with open(os.path.join(args.dir, 'score.json'), 'r') as f:
        score = json.load(f)

    last = None
    largest_t = -1
    for entry in score:
        if last is None:
            print('first entry', entry)
        elif entry['time'] - last['time'] != 1.0:
            assert(False, 'Time between notes has to be 1.0')
        if entry['time'] > largest_t:
            largest_t = entry['time']
        last = entry

    print('Largest t:', largest_t)

    app = QApplication([])
    view = GP_Control(score, not args.quiet)
    view.show()
    app.exec_()
