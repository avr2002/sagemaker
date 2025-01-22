usage:		    ## Show usage for this Makefile
	@cat Makefile | grep -E '^[a-zA-Z_-]+:.*?## .*$$' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:	    ## Install required dependencies
	bash run.sh install

run-docker:		## Build the docker image, push to ECR, and run the container
	bash run.sh run:docker

lint:		    ## Lint the code
	bash run.sh lint

lint-ci:		## Lint the code for CI
	bash run.sh lint:ci

setup-aws:		## Setup AWS for your project
	bash run.sh setup-aws

clean:			## Clean up cache and temporary files
	bash run.sh clean

help:			## Show help
	bash run.sh help
