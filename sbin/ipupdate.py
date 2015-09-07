#!/usr/bin/env python

import urllib
import urllib2

import os
import sys
import subprocess
import httplib
import time
import yaml
import base64

import requests as req
import optparse as par
import logging as log

from xml.dom.minidom import getDOMImplementation
from xml.dom.minidom import parseString


class SmartPlug(object):


    def __init__(self, host, auth):
        self.url = "http://%s:10000/smartplug.cgi" % host
        self.auth = auth
        self.domi = getDOMImplementation()

    def __xml_cmd(self, cmdId, cmdTag, cmdStr):
        doc = self.domi.createDocument(None, "SMARTPLUG", None)
        attribute = doc.documentElement.setAttribute("id", "edimax")

        cmd = doc.createElement("CMD")
        cmd.setAttribute("id", cmdId)
        # state = doc.createElement("Device.System.Power.State")
        state = doc.createElement(cmdTag)
        # state = doc.createElement("NOW_POWER")
        cmd.appendChild(state)
        state.appendChild(doc.createTextNode(cmdStr))

        doc.documentElement.appendChild(cmd)

        return doc.toxml()

    def __post_xml(self, xml):
        files = {'file': xml}

        res = req.post(self.url, auth=self.auth, files=files)

        if res.status_code == req.codes.ok:
            dom = parseString(res.text)

            try:
                val = dom.getElementsByTagName("CMD")[0].firstChild.nodeValue

                if val is None:
                    val = dom.getElementsByTagName("CMD")[0].getElementsByTagName("Device.System.Power.State")[0]. \
                        firstChild.nodeValue

                return val

            except Exception as e:

                print(e.__str__())

        return None

    @property
    def state(self):
        # res = self.__post_xml(self.__xml_cmd("get", ""))
        res = self.__post_xml(self.__xml_cmd("get", "Device.System.Power.State", ""))

        if res != "ON" and res != "OFF":
            raise Exception("Failed to communicate with SmartPlug")

        return res

    def powerInfo(self):
        xml = self.__xml_cmd("get", "NOW_POWER", "")

        files = {'file': xml}

        res = req.post(self.url, auth=self.auth, files=files)

        if res.status_code == req.codes.ok:
            # print res.text
            dom = parseString(res.text)

            try:
                cmdNode = dom.getElementsByTagName("CMD")[0]
                power = cmdNode.getElementsByTagName("Device.System.Power.NowPower")[0].firstChild.nodeValue
                current = cmdNode.getElementsByTagName("Device.System.Power.NowCurrent")[0].firstChild.nodeValue
                lastToggled = cmdNode.getElementsByTagName("Device.System.Power.LastToggleTime")[0].firstChild.nodeValue

                return [power, current, lastToggled]

            except Exception as e:
                print(e.__str__())

        return res

    @state.setter
    def state(self, value):
        if value == "ON" or value == "on":
            res = self.__post_xml(self.__xml_cmd("setup", "Device.System.Power.State", "ON"))
        else:
            res = self.__post_xml(self.__xml_cmd("setup", "Device.System.Power.State", "OFF"))

        if res != "OK":
            raise Exception("Failed to communicate with SmartPlug")


def getLocalIP():
    command = "/sbin/ip route | awk '/default/ { print $5 }'"
    locadefaultIfacelIP = subprocess.Popen(command,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           stdin=subprocess.PIPE,
                                           shell=True, executable="/bin/bash").communicate()[0]

    locadefaultIfacelIP = locadefaultIfacelIP.strip()
    command = "/sbin/ifconfig " + locadefaultIfacelIP + " | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}'"
    # print("Default interface: "), locadefaultIfacelIP
    localIP = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               stdin=subprocess.PIPE,
                               shell=True, executable="/bin/bash").communicate()[0]

    localIP = localIP.strip()
    # print("Default IP: "), localIP

    return localIP


def getNvidiaState():
    command = "/usr/bin/nvidia-smi"
    nvidiaState = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   stdin=subprocess.PIPE,
                                   shell=True, executable="/bin/bash").communicate()[0]

    #print nvidiaState
    return nvidiaState.strip()


if __name__ == "__main__":

    plug_ip = "192.168.0.103"
    plug_login = "admin"
    plug_passwd = "abcd"
    id = "workhorse"
    delay = 10

    while True:

        try:
            ns = getNvidiaState()
            nsb64 = base64.b64encode(ns);
            #print nsb64

            p = SmartPlug(plug_ip, (plug_login, plug_passwd))
            power = p.powerInfo()
            ip = getLocalIP()

            print "Updating: Power=" + power[0] + " A=" + power[1] + " ip=" + ip + " last=" + power[2]
            sys.stdout.flush()
            # url = "http://www.fajtl.net/har/update_access.php?update=" + id + "&ip=" + ip + "&power=" + power[
            #     0] + "&current=" + power[1] + "&last=" + power[2]+"&nvs="+nsb64
            # print url
            # urllib2.urlopen(url).read()

            url = 'http://www.fajtl.net/har/update_access.php'
            values = { 'update': id,
                       'ip': ip,
                       'power': power[0],
                       'current': power[1],
                       'last': power[2],
                       'nvs': nsb64,
                       }
            data = urllib.urlencode(values)

            response = urllib2.urlopen(urllib2.Request(url, data))
            result = response.read()
            if (result == "reboot"):
                print "RESTARTING"
                os.system("sudo reboot")
            if (result == "halt"):
                print "HALTING"
                os.system("sudo halt")


        except Exception as e:
            print(e.__str__())

        time.sleep(delay)
