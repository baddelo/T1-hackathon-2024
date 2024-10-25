<script>
	import '../app.css';

	import Header from '../lib/components/Header/index.svelte';
</script>

<Header />

<div class="w-11/12 d-flex mx-auto mt-2">
	<slot />
</div>
