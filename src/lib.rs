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

// impl<K: Ord, V> Ord for NodePtr<K, V> {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         unsafe { self.0.as_ref().key.cmp(&other.0.as_ref().key) }
//     }
// }

// impl<K: Ord, V> PartialOrd for NodePtr<K, V> {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         Some(self.cmp(other))
//     }
// }

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
                color: Color::Black,
                left: None,
                right: None,
                parent: None,
                key: k,
                value: v,
            })))
        })
    }

    #[inline(always)]
    fn set_color(&mut self, color: Color) {
        unsafe {
            self.0.as_mut().color = color;
        }
    }

    #[inline(always)]
    fn set_red(&mut self) {
        self.set_color(Color::Red);
    }

    #[inline(always)]
    fn set_black(&mut self) {
        self.set_color(Color::Black);
    }

    #[inline(always)]
    fn get_color(&self) -> Color {
        unsafe { self.0.as_ref().color }
    }

    #[inline(always)]
    fn is_red(&self) -> bool {
        self.get_color() == Color::Red
    }

    #[inline(always)]
    fn is_black(&self) -> bool {
        self.get_color() == Color::Black
    }

    fn get_parent(&self) -> Option<NodePtr<K, V>> {
        unsafe { self.0.as_ref().parent }
    }

    fn get_left_child(&self) -> Option<NodePtr<K, V>> {
        unsafe { self.0.as_ref().left }
    }

    fn get_right_child(&self) -> Option<NodePtr<K, V>> {
        unsafe { self.0.as_ref().right }
    }

    fn is_left_child(&self) -> bool {
        self.get_parent()
            .and_then(|parent| parent.get_left_child())
            .is_some_and(|node| node == *self)
    }

    fn is_right_child(&self) -> bool {
        self.get_parent()
            .and_then(|parent| parent.get_right_child())
            .is_some_and(|node| node == *self)
    }

    fn min_node(&self) -> Option<NodePtr<K, V>> {
        let mut p = *self;
        while let Some(left) = p.get_left_child() {
            p = left;
        }
        Some(p)
    }

    fn max_node(&self) -> Option<NodePtr<K, V>> {
        let mut p = *self;
        while let Some(node) = p.get_right_child() {
            p = node;
        }
        Some(p)
    }

    fn prev(&self) -> Option<NodePtr<K, V>> {
        todo!()
    }

    fn next(&self) -> Option<NodePtr<K, V>> {
        todo!()
    }
}

/* Test */
#[cfg(test)]
mod tests {
    use super::*;
}
