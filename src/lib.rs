//! A small ordered map backed by a red-black tree.
//!
//! [`RBTree`] stores key-value pairs ordered by key. Inserting an existing key
//! replaces the old value and returns it. Iteration yields entries in ascending
//! key order and also supports reverse traversal.
//!
//! # Examples
//!
//! ```
//! use rbtree::RBTree;
//!
//! let mut tree = RBTree::new();
//! assert_eq!(tree.insert(2, "b"), None);
//! assert_eq!(tree.insert(1, "a"), None);
//! assert_eq!(tree.insert(3, "c"), None);
//!
//! assert_eq!(tree.get(&2), Some(&"b"));
//! assert!(tree.contains_key(&1));
//!
//! let keys: Vec<_> = tree.iter().map(|(key, _)| *key).collect();
//! assert_eq!(keys, vec![1, 2, 3]);
//! ```

use std::{cmp::Ordering, fmt::Debug, marker::PhantomData, ptr::NonNull};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Color {
    Red,
    Black,
}

/* Node */
struct Node<K: Ord, V> {
    color: Color,
    parent: Option<NodePtr<K, V>>,
    left: Option<NodePtr<K, V>>,
    right: Option<NodePtr<K, V>>,
    key: K,
    value: V,
}

impl<K, V> Debug for Node<K, V>
where
    K: Ord + Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "k: {:?}\tv: {:?}\tc: {:?}",
            self.key, self.value, self.color
        )
    }
}

/* NodePtr */
#[derive(Debug)]
struct NodePtr<K: Ord, V>(NonNull<Node<K, V>>);

impl<K: Ord, V> Clone for NodePtr<K, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: Ord, V> Copy for NodePtr<K, V> {}

impl<K: Ord, V> PartialEq for NodePtr<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K: Ord, V> Eq for NodePtr<K, V> {}

impl<K: Ord, V> NodePtr<K, V> {
    fn new(k: K, v: V) -> Self {
        Self(unsafe {
            NonNull::new_unchecked(Box::into_raw(Box::new(Node {
                color: Color::Red,
                left: None,
                right: None,
                parent: None,
                key: k,
                value: v,
            })))
        })
    }

    fn set_color(&mut self, color: Color) {
        unsafe {
            self.0.as_mut().color = color;
        }
    }

    fn set_red(&mut self) {
        self.set_color(Color::Red);
    }

    fn set_black(&mut self) {
        self.set_color(Color::Black);
    }

    fn color(&self) -> Color {
        unsafe { self.0.as_ref().color }
    }

    fn is_red(&self) -> bool {
        self.color() == Color::Red
    }

    fn is_black(&self) -> bool {
        self.color() == Color::Black
    }

    fn key(&self) -> &K {
        unsafe { &self.0.as_ref().key }
    }

    fn parent(&self) -> Option<NodePtr<K, V>> {
        unsafe { self.0.as_ref().parent }
    }

    fn sibling(&self) -> Option<NodePtr<K, V>> {
        let parent = self.parent();
        if self.is_left_child() {
            parent.map(|p| p.right())?
        } else {
            parent.map(|p| p.left())?
        }
    }

    fn left(&self) -> Option<NodePtr<K, V>> {
        unsafe { self.0.as_ref().left }
    }

    fn right(&self) -> Option<NodePtr<K, V>> {
        unsafe { self.0.as_ref().right }
    }

    fn set_parent(&mut self, parent: Option<NodePtr<K, V>>) {
        unsafe { self.0.as_mut().parent = parent };
    }

    fn set_left(&mut self, left: Option<NodePtr<K, V>>) {
        unsafe { self.0.as_mut().left = left };
    }

    fn set_right(&mut self, right: Option<NodePtr<K, V>>) {
        unsafe { self.0.as_mut().right = right };
    }

    fn is_left_child(&self) -> bool {
        self.parent()
            .and_then(|parent| parent.left())
            .is_some_and(|node| node == *self)
    }

    fn is_right_child(&self) -> bool {
        self.parent()
            .and_then(|parent| parent.right())
            .is_some_and(|node| node == *self)
    }

    fn min_node(&self) -> Option<NodePtr<K, V>> {
        let mut p = *self;
        while let Some(left) = p.left() {
            p = left;
        }
        Some(p)
    }

    fn max_node(&self) -> Option<NodePtr<K, V>> {
        let mut p = *self;
        while let Some(node) = p.right() {
            p = node;
        }
        Some(p)
    }

    fn prev(&self) -> Option<NodePtr<K, V>> {
        if let Some(left) = self.left() {
            return left.max_node();
        }
        let mut p = *self;
        while let Some(parent) = p.parent() {
            if p.is_right_child() {
                return Some(parent);
            }
            p = parent;
        }
        None
    }

    fn next(&self) -> Option<NodePtr<K, V>> {
        if let Some(right) = self.right() {
            return right.min_node();
        }
        let mut p = *self;
        while let Some(parent) = p.parent() {
            if p.is_left_child() {
                return Some(parent);
            }
            p = parent;
        }
        None
    }
}

/* RBTree */

/// An ordered key-value map implemented with a red-black tree.
///
/// Keys are kept in sorted order according to their [`Ord`] implementation.
/// Operations that search by key, such as [`insert`](Self::insert),
/// [`get`](Self::get), and [`contains_key`](Self::contains_key), run in
/// logarithmic time for a balanced tree. Iteration visits every entry in key
/// order.
///
/// This type currently supports insertion, lookup, mutable lookup, clearing,
/// and ordered iteration. It does not yet provide removal of a single key.
///
/// # Examples
///
/// ```
/// use rbtree::RBTree;
///
/// let mut tree = RBTree::new();
/// tree.insert("rust", 2024);
/// tree.insert("tree", 2);
///
/// assert_eq!(tree.len(), 2);
/// assert_eq!(tree.get(&"rust"), Some(&2024));
/// ```
pub struct RBTree<K: Ord, V> {
    root: Option<NodePtr<K, V>>,
    len: usize,
    _marker: PhantomData<Box<Node<K, V>>>,
}

unsafe impl<K: Ord + Send, V: Send> Send for RBTree<K, V> {}
unsafe impl<K: Ord + Sync, V: Sync> Sync for RBTree<K, V> {}

impl<K: Ord, V> Default for RBTree<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord, V> RBTree<K, V> {
    /// Creates an empty tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbtree::RBTree;
    ///
    /// let tree: RBTree<i32, &str> = RBTree::new();
    /// assert!(tree.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            root: None,
            len: 0,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over all key-value pairs in ascending key order.
    ///
    /// The iterator borrows the tree and yields `(&K, &V)` pairs. It is also a
    /// [`DoubleEndedIterator`], so it can be reversed with [`Iterator::rev`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rbtree::RBTree;
    ///
    /// let mut tree = RBTree::new();
    /// tree.insert(3, "c");
    /// tree.insert(1, "a");
    /// tree.insert(2, "b");
    ///
    /// let forward: Vec<_> = tree.iter().map(|(k, v)| (*k, *v)).collect();
    /// assert_eq!(forward, vec![(1, "a"), (2, "b"), (3, "c")]);
    ///
    /// let backward: Vec<_> = tree.iter().rev().map(|(k, v)| (*k, *v)).collect();
    /// assert_eq!(backward, vec![(3, "c"), (2, "b"), (1, "a")]);
    /// ```
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            head: self.root.and_then(|x| x.min_node()),
            tail: self.root.and_then(|x| x.max_node()),
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns the number of key-value pairs in the tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbtree::RBTree;
    ///
    /// let mut tree = RBTree::new();
    /// assert_eq!(tree.len(), 0);
    ///
    /// tree.insert("a", 1);
    /// assert_eq!(tree.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the tree contains no key-value pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbtree::RBTree;
    ///
    /// let mut tree = RBTree::new();
    /// assert!(tree.is_empty());
    ///
    /// tree.insert(1, "one");
    /// assert!(!tree.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    fn color_of(node: Option<NodePtr<K, V>>) -> Color {
        node.map_or(Color::Black, |node| node.color())
    }

    #[inline(always)]
    fn is_black_node(node: Option<NodePtr<K, V>>) -> bool {
        Self::color_of(node) == Color::Black
    }

    #[inline(always)]
    fn is_red_node(node: Option<NodePtr<K, V>>) -> bool {
        Self::color_of(node) == Color::Red
    }

    fn find_node(&self, key: &K) -> Option<NodePtr<K, V>> {
        let mut p = self.root;
        while let Some(node) = p {
            match key.cmp(node.key()) {
                Ordering::Less => p = node.left(),
                Ordering::Greater => p = node.right(),
                Ordering::Equal => return Some(node),
            }
        }
        None
    }

    /// Returns `true` if the tree contains `key`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbtree::RBTree;
    ///
    /// let mut tree = RBTree::new();
    /// tree.insert("answer", 42);
    ///
    /// assert!(tree.contains_key(&"answer"));
    /// assert!(!tree.contains_key(&"missing"));
    /// ```
    pub fn contains_key(&self, key: &K) -> bool {
        self.find_node(key).is_some()
    }

    /// Returns a shared reference to the value for `key`.
    ///
    /// Returns [`None`] when the key is not present.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbtree::RBTree;
    ///
    /// let mut tree = RBTree::new();
    /// tree.insert(1, "one");
    ///
    /// assert_eq!(tree.get(&1), Some(&"one"));
    /// assert_eq!(tree.get(&2), None);
    /// ```
    pub fn get(&self, key: &K) -> Option<&V> {
        unsafe { self.find_node(key).map(|x| &(*x.0.as_ptr()).value) }
    }

    /// Returns a mutable reference to the value for `key`.
    ///
    /// Returns [`None`] when the key is not present.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbtree::RBTree;
    ///
    /// let mut tree = RBTree::new();
    /// tree.insert("count", 1);
    ///
    /// if let Some(value) = tree.get_mut(&"count") {
    ///     *value += 1;
    /// }
    ///
    /// assert_eq!(tree.get(&"count"), Some(&2));
    /// ```
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        unsafe { self.find_node(key).map(|x| &mut (*x.0.as_ptr()).value) }
    }

    /// Inserts a key-value pair into the tree.
    ///
    /// If the key was not present, the pair is inserted and [`None`] is
    /// returned. If the key already existed, the old value is replaced and
    /// returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbtree::RBTree;
    ///
    /// let mut tree = RBTree::new();
    ///
    /// assert_eq!(tree.insert("lang", "Rust"), None);
    /// assert_eq!(tree.insert("lang", "rust"), Some("Rust"));
    /// assert_eq!(tree.get(&"lang"), Some(&"rust"));
    /// assert_eq!(tree.len(), 1);
    /// ```
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let mut p = self.root;
        let mut parent = None;
        while let Some(mut node) = p {
            parent = Some(node);
            match &key.cmp(node.key()) {
                Ordering::Less => p = node.left(),
                Ordering::Greater => p = node.right(),
                Ordering::Equal => {
                    let old = unsafe { std::mem::replace(&mut node.0.as_mut().value, value) };
                    return Some(old);
                }
            }
        }

        let mut new = NodePtr::new(key, value);
        new.set_parent(parent);

        if let Some(mut parent) = parent {
            match new.key().cmp(parent.key()) {
                Ordering::Less => {
                    parent.set_left(Some(new));
                }
                _ => {
                    parent.set_right(Some(new));
                }
            }
        } else {
            self.root = Some(new);
        }

        self.len += 1;
        self.insert_fixup(new);

        None
    }

    fn insert_fixup(&mut self, mut node: NodePtr<K, V>) {
        while let Some(mut parent) = node.parent() {
            // Case 1: parent is black
            if parent.is_black() {
                break;
            }

            let Some(mut grand) = parent.parent() else {
                break;
            };

            let uncle = parent.sibling();
            match uncle {
                Some(mut uncle) if uncle.is_red() => {
                    // Case 2: parent is red and uncle is red
                    parent.set_black();
                    uncle.set_black();
                    grand.set_red();
                    node = grand;
                }
                _ => {
                    // Case 3: parent is red and uncle is black/nil
                    if parent.is_left_child() {
                        if node.is_right_child() {
                            // LR
                            node = parent;
                            self.rotate_left(node);
                            parent = node.parent().unwrap();
                        }
                        // LL
                        parent.set_black();
                        grand.set_red();
                        self.rotate_right(grand);
                    } else {
                        if node.is_left_child() {
                            // RL
                            node = parent;
                            self.rotate_right(node);
                            parent = node.parent().unwrap();
                        }
                        // RR
                        parent.set_black();
                        grand.set_red();
                        self.rotate_left(grand);
                    }
                }
            }
        }

        if let Some(mut root) = self.root {
            root.set_black();
        }
    }

    #[inline(always)]
    fn transplant(&mut self, x: NodePtr<K, V>, y: Option<NodePtr<K, V>>) {
        let parent = x.parent();
        if let Some(mut parent) = parent {
            if x.is_left_child() {
                parent.set_left(y);
            } else {
                parent.set_right(y);
            }
        } else {
            self.root = y;
        }
        if let Some(mut y) = y {
            y.set_parent(parent);
        }
    }

    /// Removes a key-value pair from the tree.
    ///
    /// If the key was not present, [`None`] is returned. If the key already
    /// existed, the old value is removed and returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbtree::RBTree;
    ///
    /// let mut tree = RBTree::new();
    ///
    /// tree.insert(1, 2);
    /// tree.insert(2, 4);
    /// tree.insert(3, 6);
    ///
    /// assert_eq!(tree.remove(3), Some(6));
    /// assert_eq!(tree.get(&3), None);
    /// assert_eq!(tree.len(), 2);
    /// ```
    pub fn remove(&mut self, key: K) -> Option<V> {
        let Some(node) = self.find_node(&key) else {
            return None;
        };

        // y may violate the red-black invariants
        let mut y = node;
        let mut y_original_color = y.color();

        // x is the 'double black' node
        // x can be nil, so `x_parent` and `x_is_left` is needed
        let x;
        let x_parent;
        let x_is_left;

        if node.left().is_none() {
            x = node.right();
            x_parent = node.parent();
            x_is_left = node.is_left_child();
            self.transplant(node, x);
        } else if node.right().is_none() {
            x = node.left();
            x_parent = node.parent();
            x_is_left = node.is_left_child();
            self.transplant(node, x);
        } else {
            // node logically has right child (or successor)
            // branch `node.left().is_none()` includes the situation that node has no child
            y = node.next().unwrap();
            y_original_color = y.color();
            x = y.right();

            if y.parent() == Some(node) {
                x_parent = Some(y);
                x_is_left = false;
            } else {
                x_parent = y.parent();
                x_is_left = true;

                self.transplant(y, x);
                y.set_right(node.right());

                // node has right child thus y.right() is Some
                y.right().unwrap().set_parent(Some(y));
            }

            self.transplant(node, Some(y));
            y.set_left(node.left());
            // node has left child thus y.left() is Some
            y.left().unwrap().set_parent(Some(y));
            y.set_color(node.color());
        }

        self.len -= 1;

        if y_original_color == Color::Black {
            self.remove_fixup(x, x_parent, x_is_left);
        }

        unsafe {
            let Node {
                value: old_value, ..
            } = *Box::from_raw(node.0.as_ptr());
            Some(old_value)
        }
    }

    fn remove_fixup(
        &mut self,
        mut x: Option<NodePtr<K, V>>,
        mut x_parent: Option<NodePtr<K, V>>,
        mut x_is_left: bool,
    ) {
        // the fixup ends when x is the root or it's real color is red
        while x != self.root && Self::is_black_node(x) {
            if x_is_left {
                // w is the sibling of x
                let mut w = x_parent.unwrap().right();
                if Self::is_red_node(w) {
                    // Case 1: the sibling of x is red
                    // x_parent is some
                    x_parent.unwrap().set_red();
                    if let Some(mut w) = w {
                        w.set_black();
                    }
                    self.rotate_left(x_parent.unwrap());
                    w = x_parent.unwrap().right();
                }
                let w_left = w.and_then(|w| w.left());
                let mut w_right = w.and_then(|w| w.right());
                if Self::is_black_node(w_left) && Self::is_black_node(w_right) {
                    // Case 2, sibling is black and both sibling children is black
                    if let Some(mut w) = w {
                        w.set_red();
                    }
                    x = x_parent;
                    x_parent = x.and_then(|x| x.parent());
                    x_is_left = x.map_or(false, |x| x.is_left_child())
                } else {
                    if Self::is_red_node(w_left) && Self::is_black_node(w_right) {
                        // Case 3, sibling is black and sibling left child is red, right is black
                        if let Some(mut w_left) = w_left {
                            w_left.set_black();
                        }
                        if let Some(mut w) = w {
                            w.set_red();
                            self.rotate_right(w);
                        }
                        w = x_parent.unwrap().right();
                        w_right = w.and_then(|w| w.right());
                    }
                    // Case 4, sibling is black and sibling right child is red
                    if let Some(mut w) = w {
                        w.set_color(Self::color_of(x_parent));
                    }
                    if let Some(mut w_right) = w_right {
                        w_right.set_black();
                    }
                    if let Some(mut x_parent) = x_parent {
                        x_parent.set_black();
                        self.rotate_left(x_parent);
                    }
                    x = self.root;
                }
            } else {
                // Mirror: w is the sibling of x
                let mut w = x_parent.unwrap().left();
                if Self::is_red_node(w) {
                    // Mirror Case 1: the sibling of x is red
                    // x_parent is some
                    x_parent.unwrap().set_red();
                    if let Some(mut w) = w {
                        w.set_black();
                    }
                    self.rotate_right(x_parent.unwrap());
                    w = x_parent.unwrap().left();
                }
                let w_right = w.and_then(|w| w.right());
                let mut w_left = w.and_then(|w| w.left());
                if Self::is_black_node(w_right) && Self::is_black_node(w_left) {
                    // Mirror Case 2, sibling is black and both sibling children is black
                    if let Some(mut w) = w {
                        w.set_red();
                    }
                    x = x_parent;
                    x_parent = x.and_then(|x| x.parent());
                    x_is_left = x.map_or(false, |x| x.is_left_child())
                } else {
                    if Self::is_red_node(w_right) && Self::is_black_node(w_left) {
                        // Mirror Case 3, sibling is black and sibling right child is red, left is black
                        if let Some(mut w_right) = w_right {
                            w_right.set_black();
                        }
                        if let Some(mut w) = w {
                            w.set_red();
                            self.rotate_left(w);
                        }
                        w = x_parent.unwrap().left();
                        w_left = w.and_then(|w| w.left());
                    }
                    // Case 4, sibling is black and sibling left child is red
                    if let Some(mut w) = w {
                        w.set_color(Self::color_of(x_parent));
                    }
                    if let Some(mut w_left) = w_left {
                        w_left.set_black();
                    }
                    if let Some(mut x_parent) = x_parent {
                        x_parent.set_black();
                        self.rotate_right(x_parent);
                    }
                    x = self.root;
                }
            }
        }

        if let Some(mut x) = x {
            x.set_black();
        }
        if let Some(mut root) = self.root {
            root.set_black();
        }
    }

    fn rotate_left(&mut self, mut node: NodePtr<K, V>) {
        let Some(mut right) = node.right() else {
            return;
        };
        let right_left = right.left();

        node.set_right(right_left);
        if let Some(mut right_left) = right_left {
            right_left.set_parent(Some(node));
        }

        let parent = node.parent();
        right.set_parent(parent);
        if let Some(mut parent) = parent {
            if node.is_left_child() {
                parent.set_left(Some(right));
            } else {
                parent.set_right(Some(right));
            }
        } else {
            self.root = Some(right);
        }

        right.set_left(Some(node));
        node.set_parent(Some(right));
    }

    fn rotate_right(&mut self, mut node: NodePtr<K, V>) {
        let Some(mut left) = node.left() else {
            return;
        };
        let left_right = left.right();

        node.set_left(left_right);
        if let Some(mut left_right) = left_right {
            left_right.set_parent(Some(node));
        }

        let parent = node.parent();
        left.set_parent(parent);
        if let Some(mut parent) = parent {
            if node.is_right_child() {
                parent.set_right(Some(left));
            } else {
                parent.set_left(Some(left));
            }
        } else {
            self.root = Some(left);
        }

        left.set_right(Some(node));
        node.set_parent(Some(left));
    }

    /// Removes all key-value pairs from the tree.
    ///
    /// The tree remains usable after it is cleared.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbtree::RBTree;
    ///
    /// let mut tree = RBTree::new();
    /// tree.insert(1, "a");
    /// tree.insert(2, "b");
    ///
    /// tree.clear();
    ///
    /// assert!(tree.is_empty());
    /// assert_eq!(tree.get(&1), None);
    /// ```
    pub fn clear(&mut self) {
        let mut stack = Vec::new();
        if let Some(root) = self.root.take() {
            stack.push(root);
        }

        while let Some(node) = stack.pop() {
            if let Some(left) = node.left() {
                stack.push(left);
            }
            if let Some(right) = node.right() {
                stack.push(right);
            }
            unsafe {
                drop(Box::from_raw(node.0.as_ptr()));
            }
        }

        self.len = 0;
    }
}

impl<K: Ord, V> Drop for RBTree<K, V> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<K: Ord + Debug, V: Debug> Debug for RBTree<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (k, v) in self.iter() {
            write!(f, "({:?}, {:?})", k, v)?;
        }
        write!(f, "]")
    }
}

/* Iter */
/// An iterator over borrowed entries in an [`RBTree`].
///
/// This iterator is created by [`RBTree::iter`]. It yields key-value pairs in
/// ascending key order and can also iterate from the back.
pub struct Iter<'a, K: Ord + 'a, V: 'a> {
    head: Option<NodePtr<K, V>>,
    tail: Option<NodePtr<K, V>>,
    len: usize,
    _marker: PhantomData<&'a NodePtr<K, V>>,
}

impl<'a, K: Ord + 'a, V: 'a> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        let p = self.head?;
        self.len -= 1;

        if self.len == 0 {
            self.head = None;
            self.tail = None;
        } else {
            self.head = p.next();
        }

        unsafe {
            let k = &p.0.as_ref().key;
            let v = &p.0.as_ref().value;
            Some((k, v))
        }
    }
}

impl<'a, K: Ord + 'a, V: 'a> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let p = self.tail?;
        self.len -= 1;

        if self.len == 0 {
            self.head = None;
            self.tail = None;
        } else {
            self.tail = p.prev();
        }

        unsafe {
            let k = &p.0.as_ref().key;
            let v = &p.0.as_ref().value;
            Some((k, v))
        }
    }
}

/* Test */
#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use std::{
        collections::BTreeMap,
        sync::{Arc, Mutex},
        thread::{self},
    };

    fn assert_matches_btree_map(
        tree: &RBTree<i32, i32>,
        map: &BTreeMap<i32, i32>,
        seed: u64,
        step: i32,
    ) {
        assert_eq!(tree.len(), map.len(), "seed {seed:#x}, step {step}");
        assert_eq!(
            tree.is_empty(),
            map.is_empty(),
            "seed {seed:#x}, step {step}"
        );

        let tree_entries: Vec<_> = tree.iter().map(|(key, value)| (*key, *value)).collect();
        let map_entries: Vec<_> = map.iter().map(|(key, value)| (*key, *value)).collect();
        assert_eq!(
            tree_entries, map_entries,
            "forward iter mismatch, seed {seed:#x}, step {step}"
        );

        let tree_rev_entries: Vec<_> = tree
            .iter()
            .rev()
            .map(|(key, value)| (*key, *value))
            .collect();
        let map_rev_entries: Vec<_> = map
            .iter()
            .rev()
            .map(|(key, value)| (*key, *value))
            .collect();
        assert_eq!(
            tree_rev_entries, map_rev_entries,
            "reverse iter mismatch, seed {seed:#x}, step {step}"
        );

        for key in -60..=60 {
            assert_eq!(
                tree.get(&key).copied(),
                map.get(&key).copied(),
                "get mismatch, seed {seed:#x}, step {step}, key {key}"
            );
            assert_eq!(
                tree.contains_key(&key),
                map.contains_key(&key),
                "contains_key mismatch, seed {seed:#x}, step {step}, key {key}"
            );
        }
    }

    #[test]
    fn insert_get_iter() {
        let mut tree = RBTree::new();

        let data = [10, 20, 30, 15, 5, 1, 7, 25, 40];

        for x in data {
            assert_eq!(tree.insert(x, x * 10), None);
        }

        eprintln!("{:?}", tree);

        assert_eq!(tree.len(), 9);
        assert_eq!(tree.get(&15), Some(&150));
        assert_eq!(tree.get(&99), None);

        assert_eq!(tree.insert(15, 151), Some(150));
        assert_eq!(tree.get(&15), Some(&151));
        assert_eq!(tree.len(), 9);

        let keys: Vec<_> = tree.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![1, 5, 7, 10, 15, 20, 25, 30, 40]);

        let rev_keys: Vec<_> = tree.iter().rev().map(|(k, _)| *k).collect();
        assert_eq!(rev_keys, vec![40, 30, 25, 20, 15, 10, 7, 5, 1]);
    }

    #[test]
    fn clear_tree() {
        let mut tree = RBTree::new();

        tree.insert(3, "c");
        tree.insert(1, "a");
        tree.insert(2, "b");

        assert_eq!(tree.len(), 3);

        tree.clear();

        assert!(tree.is_empty());
        assert_eq!(tree.get(&1), None);
    }

    #[test]
    fn remove_matches_btree_map_for_random_operations() {
        let seeds = [
            0x9353_8b2d_d3cf_2a9f,
            0x5d79_2e84_3f6a_4c71,
            0xc2b2_ae3d_27d4_eb4f,
            0x4f1b_92d5_7c3a_1809,
        ];

        for seed in seeds {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut tree = RBTree::new();
            let mut map = BTreeMap::new();

            for step in 0..5_000 {
                let key = rng.random_range(-50..=50);

                match rng.random_range(0..4) {
                    0 | 1 => {
                        let value = step * 17 + rng.random_range(0..17);
                        assert_eq!(
                            tree.insert(key, value),
                            map.insert(key, value),
                            "insert mismatch, seed {seed:#x}, step {step}, key {key}"
                        );
                    }
                    _ => {
                        assert_eq!(
                            tree.remove(key),
                            map.remove(&key),
                            "remove mismatch, seed {seed:#x}, step {step}, key {key}"
                        );
                    }
                }

                if step % 19 == 0 {
                    assert_matches_btree_map(&tree, &map, seed, step);
                }
            }

            assert_matches_btree_map(&tree, &map, seed, 5_000);
        }
    }

    #[test]
    fn insert_work_for_multithread() {
        let rbtree = Arc::new(Mutex::new(RBTree::new()));
        let mut handles = Vec::new();

        for i in 0..10 {
            let rbtree = rbtree.clone();
            let join_handle = thread::spawn(move || {
                rbtree.lock().unwrap().insert(i, i * i);
            });
            handles.push(join_handle);
        }

        for h in handles.into_iter() {
            let _ = h.join();
        }

        eprintln!("{:?}", rbtree.lock().unwrap());
    }
}
