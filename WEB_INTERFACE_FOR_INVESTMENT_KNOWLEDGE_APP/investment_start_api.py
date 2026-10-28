#################################################
###############   Start API   ###################
#################################################

# Import
import os
import uvicorn

# Imports
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Iniciates API server
if __name__=="__main__":
    uvicorn.run("investment_api:app", host = '0.0.0.0', port = 8000, reload = False,  workers = 3)