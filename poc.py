# -*- coding: utf-8 -*-

import pandas as pd
import torch
import logging
import util

def main():
    #test_matrix = [[1,2,3,4,5,6,7,8,9],[11,12,13,14,15,16,17,18,19],[21,22,23,24,25,26,27,28,29]]    
#    test_matrix = {"col1":[1,2,3,4,5,6,7,8,9], 
#            "col2":[11,12,13,14,15,16,17,18,19],
#            "col3":[21,22,23,24,25,26,27,28,29]
#            }
#    df = pd.DataFrame(test_matrix)
#    remove_header = df.iloc[:1,0:4]
#    print(remove_header)
    
#    x =  torch.rand(1,2,4,4)
#    print(x[0][0])
#    print(x[0][1])
#    print(x)
#    logging.basicConfig(filename="newfile.log",                     
#                    filemode='w',format='%(asctime)s %(message)s')
#    logger=logging.getLogger()
#    logger.setLevel(logging.DEBUG)
#    a = torch.randn(4, 4)
#    logger.debug("Harmless debug Message") 
#    logger.info("Just an information") 
#    logger.warning("Its a Warning") 
#    logger.error("Did you try to divide by zero") 
#    logger.critical("Internet is down")
    LOG_DIR = "./log"
    log = util.get_basic_logger(LOG_DIR,"poc")
    b = 10
    log.info("Hello {} :: ".format(str(b)))
    #print(a)
    #print(torch.argmax(a, dim=1))


if __name__ == "__main__":
    main()
