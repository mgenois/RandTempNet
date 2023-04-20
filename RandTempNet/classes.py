#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------
#-        Temporal Networks v2.0          -
#-           by Mathieu GÉNOIS            -
#-       genois.mathieu@gmail.com         -
#-  adapted in python3 by Thomas Robiglio -
#-       robigliothomas@gmail.com         -
#------------------------------------------
#Python module for handling temporal networks
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#Class definitions
#------------------------------------------
#link(): tuple of two nodes
#    link.i: int
#    link.j: int
#-Initialisation: link(i,j)
#-Methods:
#    link.display() returns (i,j), with i < j
#-Other:
#    A link() instance is hashable.
#    Comparisons between link() instances follow the rules for undirected links: (u,v) = (v,u).
class link:
    def __init__(self,i,j):
        self.i = min(i,j)
        self.j = max(i,j)
    def __eq__(self,other):
        return self.display() == other.display()
    def __ne__(self,other):
        return self.display() != other.display()
    def __hash__(self):
        return hash(self.display())
    def display(self):
        return (self.i,self.j)
#------------------------------------------
#event(): link i,j active at a time t
#    event.time: int
#    event.link: link()
#-Initialisation: event(t,i,j)
#-Methods:
#    event.change_link(i,j) updates the link with the new i,j nodes
#    event.out() returns (t,link(i,j))
#    event.display() returns (t,i,j)
class event:
    def __init__(self,t,i,j):
        self.link = link(i,j)
        self.time = t
    def __eq__(self,other):
        return self.display() == other.display()
    def __ne__(self,other):
        return self.display() != other.display()
    def __hash__(self):
        return hash(self.display())
    def change_link(self,i,j):
        self.link = link(i,j)
    def out(self):
        return (self.time,self.link)
    def display(self):
        return (self.time,self.link.i,self.link.j)
#------------------------------------------
#contact(): activation at a time t with a duration tau
#    contact.time: int
#    contact.duration: int
#-Initialisation: contact(t,tau)
#-Methods:
#    contact.out() returns (t,tau)
#    contact.display() returns (t,tau)
class contact:
    def __init__(self,t,tau):
        self.time = t
        self.duration = tau
    def __eq__(self,other):
        return self.display() == other.display()
    def __ne__(self,other):
        return self.display() != other.display()
    def __hash__(self):
        return hash(self.display())
    def display(self):
        return (self.time,self.duration)
#------------------------------------------
#-snapshot(): set of links.
#    snapshot.list_link: set of link()
#-Initialisation: snapshot(t,list_link)
#-Methods:
#    snapshot.add_link(i,j) adds a link(i,j) to list_link
#    snapshot.del_link(i,j) removes the link(i,j) from list_link
#    snapshot.add_links(list_lk) adds all links in list_lk to list_link
#    snapshot.del_links(list_lk) removes all links in list_lk from list_link
#    snapshot.out() returns set([link()...])
#    snapshot.display() returns [(i,j)...]
#-Other:
#    As list_link is a set, the unicity of links is always preserved.
class snapshot:
    def __init__(self,list_link):
        self.list_link = set(list_link)
    def add_link(self,i,j):
        self.list_link |= set([link(i,j)])
    def del_link(self,i,j):
        self.list_link -= set([link(i,j)])
    def add_links(self,list_lk):
        values = [link(i,j) for i,j in list_lk]
        self.list_link |= set(values)
    def del_links(self,list_lk):
        values = [link(i,j) for i,j in list_lk]
        self.list_link -= set(values)
    def out(self):
        return self.list_link
    def display(self):
        return [(lk.i,lk.j) for lk in self.list_link]
#------------------------------------------
#tij(): list of events ordered in time.
#    tij.data: list of events
#-Initialisation: tij()
#-Methods:
#    tij.add_event(t,i,j) adds an event (t,i,j)
#    tij.del_event(t,i,j) removes the event (t,i,j)
#    tij.add_events(list_e) adds an all events in list_e
#    tij.del_events(list_e) removes all events in list_e
#    tij.out() returns [event(),...] sorted by time
#    tij.display() returns [(t,i,j),...] sorted by time
class tij:
    def __init__(self):
        self.data = set([])
    def add_event(self,t,i,j):
        self.data |= set([event(t,i,j)])
    def del_event(self,t,i,j):
        self.data -= set([event(t,i,j)])
    def add_events(self,list_e):
        values = [event(t,i,j) for t,i,j in list_e]
        self.data |= set(values)
    def del_events(self,list_e):
        values = [event(t,i,j) for t,i,j in list_e]
        self.data -= set(values)
    def out(self):
        values = sorted(list(self.data),key=lambda x:x.time)
        return values
    def display(self):
        values = self.out()
        return [e.display() for e in values]
#------------------------------------------
#tijtau(): dictionary of durations tau labeled by triplets (t,i,j)
#    tijtau.data: dictionary of contact durations with event as key
#-Initialisation: tijtau()
#-Methods:
#    tijtau.add_contact(t,i,j,tau) adds a contact (t,i,j,tau)
#    tijtau.del_contact(t,i,j) removes the contact identified by (t,i,j)
#    tijtau.out() returns [contact(),...] sorted by time
#    tijtau.display() returns [(t,i,j,tau),...]
class tijtau:
    def __init__(self):
        self.data = {}
    def add_contact(self,t,i,j,tau):
        self.data[event(t,i,j)] = tau
    def del_contact(self,t,i,j): 
        del self.data[event(t,i,j)]
    def add_contacts(self,list_c):
        for t,i,j,tau in list_c:
            self.data[event(t,i,j)] = tau
    def del_contacts(self,list_c):
        for t,i,j in list_c:
            del self.data[event(t,i,j)]
    def out(self):
        values = [(e,self.data[e]) for e in list(self.data.keys())]
        values = sorted(values,key=lambda x:x[0].time)
        return values
    def display(self):
        values = self.out()
        return [(e.time,e.link.i,e.link.j,tau) for e,tau in values]
#------------------------------------------
#snapshot_sequence(): dictionary of snapshots ordered in time
#    snapshot_sequence.data: dictionary of snapshots with time as keys
#-Initialisation: snapshot_sequence(t_i,t_f,dt)
#-Methods:
#    snapshot_sequence.update_snapshot(t,list_link) updates the snapshot at time t with the newlist_link
#    snapshot_sequence.clear_snapshot(t) removes all the links from the snapshot at time t
#    snapshot_sequence.out() returns a list of tuples (t,list_link)
#    snapshot_sequence.display() returns [(t_i,[(i,j)...]...)
class snapshot_sequence:
    def __init__(self,t_i,t_f,dt):
        self.data = {t:snapshot([]) for t in range(t_i,t_f,dt)}
    def update_snapshot(self,t,list_link):
        self.data[t] = snapshot(list_link)
    def clear_snapshot(self,t):
        self.data[t] = snapshot([])
    def out(self):
        list_t = list(self.data.keys())
        list_t.sort()
        return [(t,list(self.data[t].out())) for t in list_t]
    def display(self):
        list_t = list(self.data.keys())
        list_t.sort()
        return [(t,[lk.display() for lk in self.data[t].out()]) for t in list_t]
#------------------------------------------
#link_timeline(): dictionary of links with their associated timeline of contacts
#    link_timeline.data: dictionary of links with their associated list of contacts
#-Initialisation: link_timeline(<list_lk>,<list_tl>) <optional>
#-Methods:
#    link_timeline.links() returns the list of links
#    link_timeline.add_link(i,j,tl) adds a link with its timeline
#    link_timeline.del_link(i,j) removes a link
#    link_timeline.add_links(list_lk,list_tl) adds a list of links with their associated timelines
#    link_timeline.del_links(list_lk) removes a list of links
#    link_timeline.add_contact(i,j,t,tau) adds a contact to the timeline of a link
#    link_timeline.del_contact(i,j,t,tau) removes a contact from the timeline of link
#    link_timeline.out() returns a list of tuples (link(),[contact(),...])
#    link_timeline.display() returns a list of tuples ((i,j),[t...])
class link_timeline:
    def __init__(self,list_lk=[],list_tl=[]):
        if list_tl != []:
            values = list(zip(list_lk,list_tl))
            self.data = {link(lk[0],lk[1]):set([contact(t,tau) for t,tau in tl]) for lk,tl in values}
        else:
            self.data = {link(i,j):set([]) for i,j in list_lk}
    def links(self):
        return list(self.data.keys())
    def links_display(self):
        values = self.links()
        return [lk.display() for lk in values]
    def add_link(self,i,j,timeline=[]):
        self.data[link(i,j)] = set([contact(t,tau) for t,tau in timeline])
    def del_link(self,i,j):
        del self.data[link(i,j)]
    def add_links(self,list_lk,list_tl=[]):
        if list_tl != []:
            values = list(zip(list_lk,list_tl))
            self.data = {link(lk[0],lk[1]):set([contact(t,tau) for t,tau in tl]) for lk,tl in values}
        else:
            self.data = {link(i,j):set([]) for i,j in list_lk}
    def del_links(self,list_lk):
        for i,j in list_lk:
            del self.data[link(i,j)]
    def add_contact(self,i,j,t,tau):
        self.data[link(i,j)] |= set([contact(t,tau)])
    def del_contact(self,i,j,t,tau):
        self.data[link(i,j)] -= set([contact(t,tau)])
    def out(self):
        return [(lk,sorted(self.data[lk],key=lambda x:x.time)) for lk in self.links()]
    def display(self):
        values = self.out()
        return [(lk.display(),[c.display() for c in tl]) for lk,tl in values]
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------