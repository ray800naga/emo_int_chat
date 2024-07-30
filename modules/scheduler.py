from transformers import get_scheduler

def my_get_scheduler(scheduler_name, optimizer, num_warmup_steps, num_training_steps):
	if scheduler_name == 'linear':
		return get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
	elif scheduler_name == 'cosine':
		return get_scheduler(name='cosine', optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
	elif scheduler_name == 'cosine_with_restarts':
		return get_scheduler(name='cosine_with_restarts', optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, scheduler_specific_kwargs={"num_cycles": 5})
	elif scheduler_name == 'polynomial':
		return get_scheduler(name='polynomial', optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, scheduler_specific_kwargs={"power": 5})
	elif scheduler_name == 'constant_with_warmup':
		return get_scheduler(name='constant_with_warmup', optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
	elif scheduler_name == 'reduce_lr_on_plateau':
		return get_scheduler(name='reduce_lr_on_plateau', optimizer=optimizer, scheduler_specific_kwargs={"mode": 'min', "factor": 0.1, "patience": 2, "verbose": True})