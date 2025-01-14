usage:		    ## Show usage for this Makefile
	@cat Makefile | grep -E '^[a-zA-Z_-]+:.*?## .*$$' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:	    ## Install required dependencies
	bash run.sh install


setup-aws:		## Setup AWS for your project
	bash run.sh setup-aws

clean:			## Clean up cache and temporary files
	bash run.sh clean

help:			## Show help
	bash run.sh help
