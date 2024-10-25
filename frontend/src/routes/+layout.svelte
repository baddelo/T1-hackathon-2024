<script>
	import '../app.css';

	import Header from '../lib/components/Header/index.svelte';
</script>

<Header />
<div class="w-10/12 flex mx-auto justify-center mt-3 pb-6">
	<slot />
</div>
