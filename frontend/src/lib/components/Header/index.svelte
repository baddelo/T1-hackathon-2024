<script>
	import { Navbar, NavBrand, NavHamburger, NavUl, NavLi } from 'flowbite-svelte';
</script>

<Navbar>
  <NavBrand class="px-3 mt-2">
    <img src="logo.svg" class="me-3 h-6 sm:h-9" alt="BadDocs Logo" />
    <span class="self-center whitespace-nowrap text-xl font-semibold">
			BadDocs
		</span>
  </NavBrand>
  <!-- <NavHamburger  />
  <NavUl >
    <NavLi href="/">Home</NavLi>
    <NavLi href="/contact">Contact</NavLi>
  </NavUl> -->
</Navbar>