# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 09:45:51 2022

@author: user
"""

import time
import modbus_tk
import modbus_tk.defines as cst
import modbus_tk.modbus_tcp as modbus_tcp

logger = modbus_tk.utils.create_logger("console")

if __name__ == "__main__":
    try:
        # 連接MODBUS TCP從機
        master = modbus_tcp.TcpMaster(host="127.0.0.1")
        master.set_timeout(5.0)
        logger.info("connected")
        
        while True:       
            logger.info(master.execute(1, cst.READ_HOLDING_REGISTERS, 0, 8))
            time.sleep(1)
        
    except modbus_tk.modbus.ModbusError as e:
        logger.error("%s- Code=%d" % (e, e.get_exception_code()))