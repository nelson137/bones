(function() {var type_impls = {
"bones_framework":[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Clone-for-RefLock%3CT%3E\" class=\"impl\"><a href=\"#impl-Clone-for-RefLock%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> for RefLock&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.clone\" class=\"method trait-impl\"><a href=\"#method.clone\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html#tymethod.clone\" class=\"fn\">clone</a>(&amp;self) -&gt; RefLock&lt;T&gt;</h4></section></summary><div class='docblock'>Returns a copy of the value. <a href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html#tymethod.clone\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.clone_from\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/1.81.0/src/core/clone.rs.html#172\">source</a></span><a href=\"#method.clone_from\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html#method.clone_from\" class=\"fn\">clone_from</a>(&amp;mut self, source: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.reference.html\">&amp;Self</a>)</h4></section></summary><div class='docblock'>Performs copy-assignment from <code>source</code>. <a href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html#method.clone_from\">Read more</a></div></details></div></details>","Clone","bones_framework::prelude::piccolo::table::TableInner","bones_framework::prelude::piccolo::thread::ExecutorInner","bones_framework::prelude::piccolo::thread::ThreadInner"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Collect-for-RefLock%3CT%3E\" class=\"impl\"><a href=\"#impl-Collect-for-RefLock%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'gc, T&gt; Collect for RefLock&lt;T&gt;<div class=\"where\">where\n    T: Collect + 'gc,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.needs_trace\" class=\"method trait-impl\"><a href=\"#method.needs_trace\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a class=\"fn\">needs_trace</a>() -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>As an optimization, if this type can never hold a <code>Gc</code> pointer and <code>trace</code> is unnecessary\nto call, you may implement this method and return false. The default implementation returns\ntrue, signaling that <code>Collect::trace</code> must be called.</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.trace\" class=\"method trait-impl\"><a href=\"#method.trace\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a class=\"fn\">trace</a>(&amp;self, cc: &amp;Collection)</h4></section></summary><div class='docblock'><em>Must</em> call <code>Collect::trace</code> on all held <code>Gc</code> pointers. If this type holds inner types that\nimplement <code>Collect</code>, a valid implementation would simply call <code>Collect::trace</code> on all the\nheld values to ensure this.</div></details></div></details>","Collect","bones_framework::prelude::piccolo::table::TableInner","bones_framework::prelude::piccolo::thread::ExecutorInner","bones_framework::prelude::piccolo::thread::ThreadInner"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Debug-for-RefLock%3CT%3E\" class=\"impl\"><a href=\"#impl-Debug-for-RefLock%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/trait.Debug.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::alloc::fmt::Debug\">Debug</a> for RefLock&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/trait.Debug.html\" title=\"trait bones_framework::asset::prelude::bones_utils::prelude::alloc::fmt::Debug\">Debug</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.fmt\" class=\"method trait-impl\"><a href=\"#method.fmt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/trait.Debug.html#tymethod.fmt\" class=\"fn\">fmt</a>(&amp;self, fmt: &amp;mut <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/struct.Formatter.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::alloc::fmt::Formatter\">Formatter</a>&lt;'_&gt;) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.81.0/core/result/enum.Result.html\" title=\"enum core::result::Result\">Result</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.unit.html\">()</a>, <a class=\"struct\" href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/struct.Error.html\" title=\"struct bones_framework::asset::prelude::bones_utils::prelude::alloc::fmt::Error\">Error</a>&gt;</h4></section></summary><div class='docblock'>Formats the value using the given formatter. <a href=\"bones_framework/asset/prelude/bones_utils/prelude/alloc/fmt/trait.Debug.html#tymethod.fmt\">Read more</a></div></details></div></details>","Debug","bones_framework::prelude::piccolo::table::TableInner","bones_framework::prelude::piccolo::thread::ExecutorInner","bones_framework::prelude::piccolo::thread::ThreadInner"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Default-for-RefLock%3CT%3E\" class=\"impl\"><a href=\"#impl-Default-for-RefLock%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a> for RefLock&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.default\" class=\"method trait-impl\"><a href=\"#method.default\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/default/trait.Default.html#tymethod.default\" class=\"fn\">default</a>() -&gt; RefLock&lt;T&gt;</h4></section></summary><div class='docblock'>Returns the “default value” for a type. <a href=\"https://doc.rust-lang.org/1.81.0/core/default/trait.Default.html#tymethod.default\">Read more</a></div></details></div></details>","Default","bones_framework::prelude::piccolo::table::TableInner","bones_framework::prelude::piccolo::thread::ExecutorInner","bones_framework::prelude::piccolo::thread::ThreadInner"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-From%3CRefCell%3CT%3E%3E-for-RefLock%3CT%3E\" class=\"impl\"><a href=\"#impl-From%3CRefCell%3CT%3E%3E-for-RefLock%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.81.0/core/cell/struct.RefCell.html\" title=\"struct core::cell::RefCell\">RefCell</a>&lt;T&gt;&gt; for RefLock&lt;T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.from\" class=\"method trait-impl\"><a href=\"#method.from\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/convert/trait.From.html#tymethod.from\" class=\"fn\">from</a>(cell: <a class=\"struct\" href=\"https://doc.rust-lang.org/1.81.0/core/cell/struct.RefCell.html\" title=\"struct core::cell::RefCell\">RefCell</a>&lt;T&gt;) -&gt; RefLock&lt;T&gt;</h4></section></summary><div class='docblock'>Converts to this type from the input type.</div></details></div></details>","From<RefCell<T>>","bones_framework::prelude::piccolo::table::TableInner","bones_framework::prelude::piccolo::thread::ExecutorInner","bones_framework::prelude::piccolo::thread::ThreadInner"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-From%3CT%3E-for-RefLock%3CT%3E\" class=\"impl\"><a href=\"#impl-From%3CT%3E-for-RefLock%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;T&gt; for RefLock&lt;T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.from\" class=\"method trait-impl\"><a href=\"#method.from\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/convert/trait.From.html#tymethod.from\" class=\"fn\">from</a>(t: T) -&gt; RefLock&lt;T&gt;</h4></section></summary><div class='docblock'>Converts to this type from the input type.</div></details></div></details>","From<T>","bones_framework::prelude::piccolo::table::TableInner","bones_framework::prelude::piccolo::thread::ExecutorInner","bones_framework::prelude::piccolo::thread::ThreadInner"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Ord-for-RefLock%3CT%3E\" class=\"impl\"><a href=\"#impl-Ord-for-RefLock%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.Ord.html\" title=\"trait core::cmp::Ord\">Ord</a> for RefLock&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.Ord.html\" title=\"trait core::cmp::Ord\">Ord</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.cmp\" class=\"method trait-impl\"><a href=\"#method.cmp\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.Ord.html#tymethod.cmp\" class=\"fn\">cmp</a>(&amp;self, other: &amp;RefLock&lt;T&gt;) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/enum.Ordering.html\" title=\"enum core::cmp::Ordering\">Ordering</a></h4></section></summary><div class='docblock'>This method returns an <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/enum.Ordering.html\" title=\"enum core::cmp::Ordering\"><code>Ordering</code></a> between <code>self</code> and <code>other</code>. <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.Ord.html#tymethod.cmp\">Read more</a></div></details></div></details>","Ord","bones_framework::prelude::piccolo::table::TableInner","bones_framework::prelude::piccolo::thread::ExecutorInner","bones_framework::prelude::piccolo::thread::ThreadInner"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-PartialEq-for-RefLock%3CT%3E\" class=\"impl\"><a href=\"#impl-PartialEq-for-RefLock%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialEq.html\" title=\"trait core::cmp::PartialEq\">PartialEq</a> for RefLock&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialEq.html\" title=\"trait core::cmp::PartialEq\">PartialEq</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.eq\" class=\"method trait-impl\"><a href=\"#method.eq\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialEq.html#tymethod.eq\" class=\"fn\">eq</a>(&amp;self, other: &amp;RefLock&lt;T&gt;) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>This method tests for <code>self</code> and <code>other</code> values to be equal, and is used\nby <code>==</code>.</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.ne\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/1.81.0/src/core/cmp.rs.html#262\">source</a></span><a href=\"#method.ne\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialEq.html#method.ne\" class=\"fn\">ne</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>This method tests for <code>!=</code>. The default implementation is almost always\nsufficient, and should not be overridden without very good reason.</div></details></div></details>","PartialEq","bones_framework::prelude::piccolo::table::TableInner","bones_framework::prelude::piccolo::thread::ExecutorInner","bones_framework::prelude::piccolo::thread::ThreadInner"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-PartialOrd-for-RefLock%3CT%3E\" class=\"impl\"><a href=\"#impl-PartialOrd-for-RefLock%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialOrd.html\" title=\"trait core::cmp::PartialOrd\">PartialOrd</a> for RefLock&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialOrd.html\" title=\"trait core::cmp::PartialOrd\">PartialOrd</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.partial_cmp\" class=\"method trait-impl\"><a href=\"#method.partial_cmp\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialOrd.html#tymethod.partial_cmp\" class=\"fn\">partial_cmp</a>(&amp;self, other: &amp;RefLock&lt;T&gt;) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.81.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;<a class=\"enum\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/enum.Ordering.html\" title=\"enum core::cmp::Ordering\">Ordering</a>&gt;</h4></section></summary><div class='docblock'>This method returns an ordering between <code>self</code> and <code>other</code> values if one exists. <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialOrd.html#tymethod.partial_cmp\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.lt\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/1.81.0/src/core/cmp.rs.html#1179\">source</a></span><a href=\"#method.lt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialOrd.html#method.lt\" class=\"fn\">lt</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>This method tests less than (for <code>self</code> and <code>other</code>) and is used by the <code>&lt;</code> operator. <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialOrd.html#method.lt\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.le\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/1.81.0/src/core/cmp.rs.html#1197\">source</a></span><a href=\"#method.le\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialOrd.html#method.le\" class=\"fn\">le</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>This method tests less than or equal to (for <code>self</code> and <code>other</code>) and is used by the <code>&lt;=</code>\noperator. <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialOrd.html#method.le\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.gt\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/1.81.0/src/core/cmp.rs.html#1214\">source</a></span><a href=\"#method.gt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialOrd.html#method.gt\" class=\"fn\">gt</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>This method tests greater than (for <code>self</code> and <code>other</code>) and is used by the <code>&gt;</code> operator. <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialOrd.html#method.gt\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.ge\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/1.81.0/src/core/cmp.rs.html#1232\">source</a></span><a href=\"#method.ge\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialOrd.html#method.ge\" class=\"fn\">ge</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>This method tests greater than or equal to (for <code>self</code> and <code>other</code>) and is used by the <code>&gt;=</code>\noperator. <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialOrd.html#method.ge\">Read more</a></div></details></div></details>","PartialOrd","bones_framework::prelude::piccolo::table::TableInner","bones_framework::prelude::piccolo::thread::ExecutorInner","bones_framework::prelude::piccolo::thread::ThreadInner"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-RefLock%3CT%3E\" class=\"impl\"><a href=\"#impl-RefLock%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; RefLock&lt;T&gt;</h3></section></summary><div class=\"impl-items\"><section id=\"method.new\" class=\"method\"><h4 class=\"code-header\">pub fn <a class=\"fn\">new</a>(t: T) -&gt; RefLock&lt;T&gt;</h4></section><section id=\"method.into_inner\" class=\"method\"><h4 class=\"code-header\">pub fn <a class=\"fn\">into_inner</a>(self) -&gt; T</h4></section><section id=\"method.take\" class=\"method\"><h4 class=\"code-header\">pub fn <a class=\"fn\">take</a>(&amp;self) -&gt; T<div class=\"where\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>,</div></h4></section></div></details>",0,"bones_framework::prelude::piccolo::table::TableInner","bones_framework::prelude::piccolo::thread::ExecutorInner","bones_framework::prelude::piccolo::thread::ThreadInner"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-RefLock%3CT%3E\" class=\"impl\"><a href=\"#impl-RefLock%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; RefLock&lt;T&gt;<div class=\"where\">where\n    T: ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><section id=\"method.as_ptr\" class=\"method\"><h4 class=\"code-header\">pub fn <a class=\"fn\">as_ptr</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.pointer.html\">*mut T</a></h4></section><section id=\"method.borrow\" class=\"method\"><h4 class=\"code-header\">pub fn <a class=\"fn\">borrow</a>&lt;'a&gt;(&amp;'a self) -&gt; <a class=\"struct\" href=\"https://doc.rust-lang.org/1.81.0/core/cell/struct.Ref.html\" title=\"struct core::cell::Ref\">Ref</a>&lt;'a, T&gt;</h4></section><section id=\"method.try_borrow\" class=\"method\"><h4 class=\"code-header\">pub fn <a class=\"fn\">try_borrow</a>&lt;'a&gt;(&amp;'a self) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.81.0/core/result/enum.Result.html\" title=\"enum core::result::Result\">Result</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.81.0/core/cell/struct.Ref.html\" title=\"struct core::cell::Ref\">Ref</a>&lt;'a, T&gt;, <a class=\"struct\" href=\"https://doc.rust-lang.org/1.81.0/core/cell/struct.BorrowError.html\" title=\"struct core::cell::BorrowError\">BorrowError</a>&gt;</h4></section><details class=\"toggle method-toggle\" open><summary><section id=\"method.as_ref_cell\" class=\"method\"><h4 class=\"code-header\">pub unsafe fn <a class=\"fn\">as_ref_cell</a>(&amp;self) -&gt; &amp;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.81.0/core/cell/struct.RefCell.html\" title=\"struct core::cell::RefCell\">RefCell</a>&lt;T&gt;</h4></section></summary><div class=\"docblock\"><p>Access the wrapped <a href=\"https://doc.rust-lang.org/1.81.0/core/cell/struct.RefCell.html\" title=\"struct core::cell::RefCell\"><code>RefCell</code></a>.</p>\n<h5 id=\"safety\"><a class=\"doc-anchor\" href=\"#safety\">§</a>Safety</h5>\n<p>In order to maintain the invariants of the garbage collector, no new [<code>Gc</code>]\npointers may be adopted by this type as a result of the interior mutability\nafforded by directly accessing the inner <a href=\"https://doc.rust-lang.org/1.81.0/core/cell/struct.RefCell.html\" title=\"struct core::cell::RefCell\"><code>RefCell</code></a>, unless the write barrier for the containing [<code>Gc</code>] pointer is invoked manuall\nbefore collection is triggered.</p>\n</div></details><section id=\"method.get_mut\" class=\"method\"><h4 class=\"code-header\">pub fn <a class=\"fn\">get_mut</a>(&amp;mut self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.reference.html\">&amp;mut T</a></h4></section></div></details>",0,"bones_framework::prelude::piccolo::table::TableInner","bones_framework::prelude::piccolo::thread::ExecutorInner","bones_framework::prelude::piccolo::thread::ThreadInner"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Unlock-for-RefLock%3CT%3E\" class=\"impl\"><a href=\"#impl-Unlock-for-RefLock%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; Unlock for RefLock&lt;T&gt;<div class=\"where\">where\n    T: ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle\" open><summary><section id=\"associatedtype.Unlocked\" class=\"associatedtype trait-impl\"><a href=\"#associatedtype.Unlocked\" class=\"anchor\">§</a><h4 class=\"code-header\">type <a class=\"associatedtype\">Unlocked</a> = <a class=\"struct\" href=\"https://doc.rust-lang.org/1.81.0/core/cell/struct.RefCell.html\" title=\"struct core::cell::RefCell\">RefCell</a>&lt;T&gt;</h4></section></summary><div class='docblock'>This will typically be a cell-like type providing some sort of interior mutability.</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.unlock_unchecked\" class=\"method trait-impl\"><a href=\"#method.unlock_unchecked\" class=\"anchor\">§</a><h4 class=\"code-header\">unsafe fn <a class=\"fn\">unlock_unchecked</a>(&amp;self) -&gt; &amp;&lt;RefLock&lt;T&gt; as Unlock&gt;::Unlocked</h4></section></summary><div class='docblock'>Provides unsafe access to the unlocked type, <em>without</em> triggering a write barrier. <a>Read more</a></div></details></div></details>","Unlock","bones_framework::prelude::piccolo::table::TableInner","bones_framework::prelude::piccolo::thread::ExecutorInner","bones_framework::prelude::piccolo::thread::ThreadInner"],["<section id=\"impl-Eq-for-RefLock%3CT%3E\" class=\"impl\"><a href=\"#impl-Eq-for-RefLock%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> for RefLock&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section>","Eq","bones_framework::prelude::piccolo::table::TableInner","bones_framework::prelude::piccolo::thread::ExecutorInner","bones_framework::prelude::piccolo::thread::ThreadInner"],["<section id=\"impl-StructuralPartialEq-for-RefLock%3CT%3E\" class=\"impl\"><a href=\"#impl-StructuralPartialEq-for-RefLock%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/marker/trait.StructuralPartialEq.html\" title=\"trait core::marker::StructuralPartialEq\">StructuralPartialEq</a> for RefLock&lt;T&gt;<div class=\"where\">where\n    T: ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h3></section>","StructuralPartialEq","bones_framework::prelude::piccolo::table::TableInner","bones_framework::prelude::piccolo::thread::ExecutorInner","bones_framework::prelude::piccolo::thread::ThreadInner"]]
};if (window.register_type_impls) {window.register_type_impls(type_impls);} else {window.pending_type_impls = type_impls;}})()