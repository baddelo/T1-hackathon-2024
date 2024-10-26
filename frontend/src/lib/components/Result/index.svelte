<script>
	import { fake_date } from './fake_data.js';

	// export let text_result;

	const date = new Date().toLocaleDateString();
	const time = new Date().toLocaleTimeString();
</script>

<div class="flex flex-col gap-3">
	<h1>{date} {time}</h1>
	<div class="flex flex-col gap-1">
		{#each fake_date as item}
			<p class="p-0">{item.content}</p>
		{/each}
	</div>
</div>