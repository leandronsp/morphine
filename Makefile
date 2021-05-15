console:
	docker-compose run dev bash

mix.deps.get:
	docker-compose run dev mix deps.get

mix.test:
	docker-compose run dev mix test
