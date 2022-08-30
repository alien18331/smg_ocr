# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:08:21 2022

@author: user
"""

import sys
import logging
import random
import threading
import modbus_tk
import modbus_tk.defines as cst
import modbus_tk.modbus as modbus
import modbus_tk.modbus_tcp as modbus_tcp
LOGGER = modbus_tk.utils.create_logger(name="console", record_format="%(message)s")

if __name__ == "__main__":
    try:
        # server裏的address需要寫的樹莓派的IP和需要開放的端口，注意開放相應的端口
        SERVER = modbus_tcp.TcpServer(address="127.0.0.1", port=502)
        LOGGER.info("running...")
        LOGGER.info("enter 'quit' for closing the server")
        # 服務啓動
        SERVER.start()
        # 建立第一個從機
        SLAVE1 = SERVER.add_slave(1)
        SLAVE1.add_block('A', cst.HOLDING_REGISTERS, 0, 8)#地址0，長度8
        # SLAVE1.add_block('B', cst.HOLDING_REGISTERS, 4, 14)

        #建立另一個從機2
        # SLAVE2 = SERVER.add_slave(2)
        # SLAVE2.add_block('C', cst.COILS, 0, 10)   #地址0，長度10
        # SLAVE2.add_block('D', cst.HOLDING_REGISTERS, 0, 10)#地址0，長度10
        
        while True:
            SLAVE1.set_values('A', 0, int(random.uniform(3.5,4.5)*100)) #改變在地址0處的寄存器的值
            # SLAVE1.set_values('B', 4, [1, 2, 3, 4, 5, 5, 12, 1232])     #改變在地址4處的寄存器的值
            # SLAVE2.set_values('C', 0, [1, 1, 1, 1, 1, 1])
            # SLAVE2.set_values('D', 0, 10)

        # while True:
        #     CMD = sys.stdin.readline()
        #     if CMD.find('quit') == 0:
        #         sys.stdout.write('bye-bye\r\n')
        #         break
        #     else:
        #         sys.stdout.write("unknown command %s\r\n" % (args[0]))
    
    except KeyboardInterrupt:    
        SERVER._do_exit()
        SERVER.stop()
        
    finally:
        SERVER.stop()