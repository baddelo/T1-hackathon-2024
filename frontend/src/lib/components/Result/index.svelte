<script>
	export let text_result;

	const date = new Date().toLocaleDateString();
	const time = new Date().toLocaleTimeString()
</script>

<div class="flex flex-col gap-3">
	<h1>{date} {time}</h1>
	<p>{text_result}</p>
</div>