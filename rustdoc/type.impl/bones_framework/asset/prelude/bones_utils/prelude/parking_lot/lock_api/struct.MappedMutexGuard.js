(function() {var type_impls = {
"bones_framework":[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"impl\"><a href=\"#impl-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a, R, T&gt; <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, T&gt;<div class=\"where\">where\n    R: <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/trait.RawMutex.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::RawMutex\">RawMutex</a> + 'a,\n    T: 'a + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.map\" class=\"method\"><h4 class=\"code-header\">pub fn <a href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html#tymethod.map\" class=\"fn\">map</a>&lt;U, F&gt;(\n    s: <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, T&gt;,\n    f: F\n) -&gt; <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, U&gt;<div class=\"where\">where\n    F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/ops/function/trait.FnOnce.html\" title=\"trait core::ops::function::FnOnce\">FnOnce</a>(<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.76.0/std/primitive.reference.html\">&amp;mut T</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.76.0/std/primitive.reference.html\">&amp;mut U</a>,\n    U: ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h4></section></summary><div class=\"docblock\"><p>Makes a new <code>MappedMutexGuard</code> for a component of the locked data.</p>\n<p>This operation cannot fail as the <code>MappedMutexGuard</code> passed\nin already locked the mutex.</p>\n<p>This is an associated function that needs to be\nused as <code>MappedMutexGuard::map(...)</code>. A method would interfere with methods of\nthe same name on the contents of the locked data.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.try_map\" class=\"method\"><h4 class=\"code-header\">pub fn <a href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html#tymethod.try_map\" class=\"fn\">try_map</a>&lt;U, F&gt;(\n    s: <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, T&gt;,\n    f: F\n) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.76.0/core/result/enum.Result.html\" title=\"enum core::result::Result\">Result</a>&lt;<a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, U&gt;, <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, T&gt;&gt;<div class=\"where\">where\n    F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/ops/function/trait.FnOnce.html\" title=\"trait core::ops::function::FnOnce\">FnOnce</a>(<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.76.0/std/primitive.reference.html\">&amp;mut T</a>) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.76.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.76.0/std/primitive.reference.html\">&amp;mut U</a>&gt;,\n    U: ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h4></section></summary><div class=\"docblock\"><p>Attempts to make a new <code>MappedMutexGuard</code> for a component of the\nlocked data. The original guard is returned if the closure returns <code>None</code>.</p>\n<p>This operation cannot fail as the <code>MappedMutexGuard</code> passed\nin already locked the mutex.</p>\n<p>This is an associated function that needs to be\nused as <code>MappedMutexGuard::try_map(...)</code>. A method would interfere with methods of\nthe same name on the contents of the locked data.</p>\n</div></details></div></details>",0,"bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedFairMutexGuard","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedMutexGuard"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"impl\"><a href=\"#impl-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a, R, T&gt; <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, T&gt;<div class=\"where\">where\n    R: <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/trait.RawMutexFair.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::RawMutexFair\">RawMutexFair</a> + 'a,\n    T: 'a + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.unlock_fair\" class=\"method\"><h4 class=\"code-header\">pub fn <a href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html#tymethod.unlock_fair\" class=\"fn\">unlock_fair</a>(s: <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, T&gt;)</h4></section></summary><div class=\"docblock\"><p>Unlocks the mutex using a fair unlock protocol.</p>\n<p>By default, mutexes are unfair and allow the current thread to re-lock\nthe mutex before another has the chance to acquire the lock, even if\nthat thread has been blocked on the mutex for a long time. This is the\ndefault because it allows much higher throughput as it avoids forcing a\ncontext switch on every mutex unlock. This can result in one thread\nacquiring a mutex many more times than other threads.</p>\n<p>However in some cases it can be beneficial to ensure fairness by forcing\nthe lock to pass on to a waiting thread if there is one. This is done by\nusing this method instead of dropping the <code>MutexGuard</code> normally.</p>\n</div></details></div></details>",0,"bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedFairMutexGuard","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedMutexGuard"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-DerefMut-for-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"impl\"><a href=\"#impl-DerefMut-for-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a, R, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/ops/deref/trait.DerefMut.html\" title=\"trait core::ops::deref::DerefMut\">DerefMut</a> for <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, T&gt;<div class=\"where\">where\n    R: <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/trait.RawMutex.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::RawMutex\">RawMutex</a> + 'a,\n    T: 'a + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.deref_mut\" class=\"method trait-impl\"><a href=\"#method.deref_mut\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.76.0/core/ops/deref/trait.DerefMut.html#tymethod.deref_mut\" class=\"fn\">deref_mut</a>(&amp;mut self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.76.0/std/primitive.reference.html\">&amp;mut T</a></h4></section></summary><div class='docblock'>Mutably dereferences the value.</div></details></div></details>","DerefMut","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedFairMutexGuard","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedMutexGuard"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Debug-for-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"impl\"><a href=\"#impl-Debug-for-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a, R, T&gt; <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/trait.Debug.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::alloc::fmt::Debug\">Debug</a> for <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, T&gt;<div class=\"where\">where\n    R: <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/trait.RawMutex.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::RawMutex\">RawMutex</a> + 'a,\n    T: <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/trait.Debug.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::alloc::fmt::Debug\">Debug</a> + 'a + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.fmt\" class=\"method trait-impl\"><a href=\"#method.fmt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/trait.Debug.html#tymethod.fmt\" class=\"fn\">fmt</a>(&amp;self, f: &amp;mut <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/struct.Formatter.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::alloc::fmt::Formatter\">Formatter</a>&lt;'_&gt;) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.76.0/core/result/enum.Result.html\" title=\"enum core::result::Result\">Result</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.76.0/std/primitive.unit.html\">()</a>, <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/struct.Error.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::alloc::fmt::Error\">Error</a>&gt;</h4></section></summary><div class='docblock'>Formats the value using the given formatter. <a href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/trait.Debug.html#tymethod.fmt\">Read more</a></div></details></div></details>","Debug","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedFairMutexGuard","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedMutexGuard"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Drop-for-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"impl\"><a href=\"#impl-Drop-for-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a, R, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/ops/drop/trait.Drop.html\" title=\"trait core::ops::drop::Drop\">Drop</a> for <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, T&gt;<div class=\"where\">where\n    R: <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/trait.RawMutex.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::RawMutex\">RawMutex</a> + 'a,\n    T: 'a + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.drop\" class=\"method trait-impl\"><a href=\"#method.drop\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.76.0/core/ops/drop/trait.Drop.html#tymethod.drop\" class=\"fn\">drop</a>(&amp;mut self)</h4></section></summary><div class='docblock'>Executes the destructor for this type. <a href=\"https://doc.rust-lang.org/1.76.0/core/ops/drop/trait.Drop.html#tymethod.drop\">Read more</a></div></details></div></details>","Drop","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedFairMutexGuard","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedMutexGuard"],["<section id=\"impl-Sync-for-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"impl\"><a href=\"#impl-Sync-for-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a, R, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> for <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, T&gt;<div class=\"where\">where\n    R: <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/trait.RawMutex.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::RawMutex\">RawMutex</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> + 'a,\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> + 'a + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section>","Sync","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedFairMutexGuard","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedMutexGuard"],["<section id=\"impl-Send-for-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"impl\"><a href=\"#impl-Send-for-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a, R, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, T&gt;<div class=\"where\">where\n    R: <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/trait.RawMutex.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::RawMutex\">RawMutex</a> + 'a,\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + 'a + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,\n    &lt;R as <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/trait.RawMutex.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::RawMutex\">RawMutex</a>&gt;::<a class=\"associatedtype\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/trait.RawMutex.html#associatedtype.GuardMarker\" title=\"type bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::RawMutex::GuardMarker\">GuardMarker</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,</div></h3></section>","Send","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedFairMutexGuard","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedMutexGuard"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Display-for-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"impl\"><a href=\"#impl-Display-for-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a, R, T&gt; <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/trait.Display.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::alloc::fmt::Display\">Display</a> for <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, T&gt;<div class=\"where\">where\n    R: <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/trait.RawMutex.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::RawMutex\">RawMutex</a> + 'a,\n    T: <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/trait.Display.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::alloc::fmt::Display\">Display</a> + 'a + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.fmt\" class=\"method trait-impl\"><a href=\"#method.fmt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/trait.Display.html#tymethod.fmt\" class=\"fn\">fmt</a>(&amp;self, f: &amp;mut <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/struct.Formatter.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::alloc::fmt::Formatter\">Formatter</a>&lt;'_&gt;) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.76.0/core/result/enum.Result.html\" title=\"enum core::result::Result\">Result</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.76.0/std/primitive.unit.html\">()</a>, <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/struct.Error.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::alloc::fmt::Error\">Error</a>&gt;</h4></section></summary><div class='docblock'>Formats the value using the given formatter. <a href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/trait.Display.html#tymethod.fmt\">Read more</a></div></details></div></details>","Display","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedFairMutexGuard","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedMutexGuard"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Deref-for-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"impl\"><a href=\"#impl-Deref-for-MappedMutexGuard%3C'a,+R,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a, R, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/ops/deref/trait.Deref.html\" title=\"trait core::ops::deref::Deref\">Deref</a> for <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/struct.MappedMutexGuard.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::MappedMutexGuard\">MappedMutexGuard</a>&lt;'a, R, T&gt;<div class=\"where\">where\n    R: <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/parking_lot/lock_api/trait.RawMutex.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::parking_lot::lock_api::RawMutex\">RawMutex</a> + 'a,\n    T: 'a + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.76.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle\" open><summary><section id=\"associatedtype.Target\" class=\"associatedtype trait-impl\"><a href=\"#associatedtype.Target\" class=\"anchor\">§</a><h4 class=\"code-header\">type <a href=\"https://doc.rust-lang.org/1.76.0/core/ops/deref/trait.Deref.html#associatedtype.Target\" class=\"associatedtype\">Target</a> = T</h4></section></summary><div class='docblock'>The resulting type after dereferencing.</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.deref\" class=\"method trait-impl\"><a href=\"#method.deref\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.76.0/core/ops/deref/trait.Deref.html#tymethod.deref\" class=\"fn\">deref</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.76.0/std/primitive.reference.html\">&amp;T</a></h4></section></summary><div class='docblock'>Dereferences the value.</div></details></div></details>","Deref","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedFairMutexGuard","bones_framework::asset::prelude::bones_utils::prelude::parking_lot::MappedMutexGuard"]]
};if (window.register_type_impls) {window.register_type_impls(type_impls);} else {window.pending_type_impls = type_impls;}})()