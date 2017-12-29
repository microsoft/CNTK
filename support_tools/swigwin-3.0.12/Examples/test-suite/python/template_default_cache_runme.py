import template_default_cache

ap = template_default_cache.get_mp_a();
bp = template_default_cache.get_mp_b();

if not isinstance(ap, template_default_cache.AModelPtr):
    raise RuntimeError("get_mp_a fail")
if not isinstance(bp, template_default_cache.BModelPtr):
    raise RuntimeError("get_mp_b fail")
