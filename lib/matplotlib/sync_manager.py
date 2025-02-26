"""
Synchronization manager for linked Matplotlib views.

This module provides a publish-subscribe architecture for coordinating view changes
across linked plots, with support for complex view relationships and efficient
handling of large view counts.
"""

import weakref
from enum import Enum, auto


class LinkType(Enum):
    """Enum defining the types of links between axes."""
    X_ONLY = auto()  # Link only the x-axis
    Y_ONLY = auto()  # Link only the y-axis
    BOTH = auto()    # Link both axes


class PropagationDirection(Enum):
    """Enum defining the direction of update propagation."""
    PARENT_TO_CHILD = auto()  # Updates flow from parent to children only
    BIDIRECTIONAL = auto()    # Updates flow in both directions


class ViewGroup:
    """
    A group of views that are linked together.
    
    Views in the same group will have their limits synchronized according
    to the link type and propagation rules.
    """
    
    def __init__(self, name=None, propagation=PropagationDirection.BIDIRECTIONAL):
        """
        Initialize a new view group.
        
        Parameters
        ----------
        name : str, optional
            A name for this view group.
        propagation : PropagationDirection, default: BIDIRECTIONAL
            The direction of update propagation within this group.
        """
        self.name = name
        self.propagation = propagation
        self._views = {}  # {view: (parent, link_type)}
        self._updating = False  # Flag to prevent infinite recursion
    
    def add_view(self, view, parent=None, link_type=LinkType.BOTH):
        """
        Add a view to this group.
        
        Parameters
        ----------
        view : matplotlib.axes.Axes
            The view to add to this group.
        parent : matplotlib.axes.Axes, optional
            The parent view. If None, this view has no parent in this group.
        link_type : LinkType, default: BOTH
            The type of link between this view and its parent.
        """
        if view in self._views and self._views[view][0] is not None:
            # View already has a parent in this group
            raise ValueError(f"View {view} already has a parent in this group")
        
        # Store weak references to avoid circular references
        view_ref = weakref.ref(view)
        parent_ref = weakref.ref(parent) if parent is not None else None
        
        self._views[view_ref] = (parent_ref, link_type)
        
        # Set up the initial synchronization
        if parent is not None:
            self._sync_view_with_parent(view, parent, link_type)
    
    def _sync_view_with_parent(self, view, parent, link_type):
        """
        Synchronize a view with its parent.
        
        Parameters
        ----------
        view : matplotlib.axes.Axes
            The view to synchronize.
        parent : matplotlib.axes.Axes
            The parent view.
        link_type : LinkType
            The type of link between the view and its parent.
        """
        if link_type in (LinkType.X_ONLY, LinkType.BOTH):
            view.set_xlim(parent.get_xlim())
        
        if link_type in (LinkType.Y_ONLY, LinkType.BOTH):
            view.set_ylim(parent.get_ylim())
    
    def update_view(self, updated_view, xlim=None, ylim=None):
        """
        Update a view and propagate the changes to linked views.
        
        Parameters
        ----------
        updated_view : matplotlib.axes.Axes
            The view that was updated.
        xlim : tuple, optional
            The new x limits. If None, the x limits are not updated.
        ylim : tuple, optional
            The new y limits. If None, the y limits are not updated.
        """
        if self._updating:
            # Prevent infinite recursion
            return
        
        self._updating = True
        try:
            # Find the view in our dictionary
            view_ref = None
            for ref in self._views:
                if ref() is updated_view:
                    view_ref = ref
                    break
            
            if view_ref is None:
                # View not in this group
                return
            
            # Update children if propagation allows
            if self.propagation != PropagationDirection.PARENT_TO_CHILD:
                # Update parent
                parent_ref, link_type = self._views[view_ref]
                if parent_ref is not None:
                    parent = parent_ref()
                    if parent is not None:
                        if xlim is not None and link_type in (LinkType.X_ONLY, LinkType.BOTH):
                            parent.set_xlim(xlim)
                        if ylim is not None and link_type in (LinkType.Y_ONLY, LinkType.BOTH):
                            parent.set_ylim(ylim)
            
            # Update children
            for child_ref, (parent_ref, link_type) in self._views.items():
                if parent_ref is None:
                    continue
                
                child = child_ref()
                parent = parent_ref()
                
                if child is None or parent is None:
                    # Weak reference expired
                    continue
                
                if parent is updated_view:
                    if xlim is not None and link_type in (LinkType.X_ONLY, LinkType.BOTH):
                        child.set_xlim(xlim)
                    if ylim is not None and link_type in (LinkType.Y_ONLY, LinkType.BOTH):
                        child.set_ylim(ylim)
        finally:
            self._updating = False


class SyncManager:
    """
    Manager for synchronizing views across multiple groups.
    
    This class provides a centralized manager for view synchronization,
    allowing views to participate in multiple groups with different
    link types and propagation rules.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the SyncManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize a new SyncManager."""
        self._groups = {}  # {name: ViewGroup}
        self._view_groups = {}  # {view: [group_name]}
    
    def create_group(self, name, propagation=PropagationDirection.BIDIRECTIONAL):
        """
        Create a new view group.
        
        Parameters
        ----------
        name : str
            A name for this view group.
        propagation : PropagationDirection, default: BIDIRECTIONAL
            The direction of update propagation within this group.
        
        Returns
        -------
        ViewGroup
            The newly created view group.
        """
        if name in self._groups:
            raise ValueError(f"Group {name} already exists")
        
        group = ViewGroup(name, propagation)
        self._groups[name] = group
        return group
    
    def get_group(self, name):
        """
        Get a view group by name.
        
        Parameters
        ----------
        name : str
            The name of the view group.
        
        Returns
        -------
        ViewGroup
            The view group with the given name.
        """
        if name not in self._groups:
            raise ValueError(f"Group {name} does not exist")
        
        return self._groups[name]
    
    def add_view_to_group(self, view, group_name, parent=None, link_type=LinkType.BOTH):
        """
        Add a view to a group.
        
        Parameters
        ----------
        view : matplotlib.axes.Axes
            The view to add to the group.
        group_name : str
            The name of the group to add the view to.
        parent : matplotlib.axes.Axes, optional
            The parent view. If None, this view has no parent in this group.
        link_type : LinkType, default: BOTH
            The type of link between this view and its parent.
        """
        group = self.get_group(group_name)
        group.add_view(view, parent, link_type)
        
        # Keep track of which groups this view belongs to
        view_ref = weakref.ref(view)
        if view_ref not in self._view_groups:
            self._view_groups[view_ref] = []
        
        if group_name not in self._view_groups[view_ref]:
            self._view_groups[view_ref].append(group_name)
    
    def update_view(self, view, xlim=None, ylim=None):
        """
        Update a view and propagate the changes to linked views.
        
        Parameters
        ----------
        view : matplotlib.axes.Axes
            The view that was updated.
        xlim : tuple, optional
            The new x limits. If None, the x limits are not updated.
        ylim : tuple, optional
            The new y limits. If None, the y limits are not updated.
        """
        # Find the view in our dictionary
        view_ref = None
        for ref in self._view_groups:
            if ref() is view:
                view_ref = ref
                break
        
        if view_ref is None:
            # View not in any group
            return
        
        # Update all groups this view belongs to
        for group_name in self._view_groups[view_ref]:
            group = self._groups[group_name]
            group.update_view(view, xlim, ylim)