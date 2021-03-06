import collections
import functools
import operator
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck

def threshold_weights_global_norm_without_dec(structure,replace_structure, thresh):
  """Does not communicate if all layers below threshold.
  Args:
    structure: A structure supported by tf.nest.
  Returns:
     A tuple (input, 1) if all entries are finite or the structure is empty, or
     a tuple (zeros, 0) if any non-finite entries were found.
  """
  flat = tf.nest.flatten(structure)
  if not flat:
    return (structure, tf.constant(0))
  client_norm = tf.linalg.global_norm(structure)
  if client_norm>thresh:
    return (structure, tf.constant(1, dtype=tf.int32), client_norm)
  else:
    #return (tf.nest.map_structure(tf.zeros_like, structure), tf.constant(0,  dtype=tf.int32))
    return (replace_structure, tf.constant(0,  dtype=tf.int32), client_norm)


@tf.function
def threshold_weights_global_norm(structure,replace_structure, thresh):
  """Does not communicate if all layers below threshold.
  Args:
    structure: A structure supported by tf.nest.
  Returns:
     A tuple (input, 1) if all entries are finite or the structure is empty, or
     a tuple (zeros, 0) if any non-finite entries were found.
  """
  flat = tf.nest.flatten(structure)
  if not flat:
    return (structure, tf.constant(0))
  client_norm = tf.linalg.global_norm(structure)
  if client_norm>thresh:
    return (structure, tf.constant(1, dtype=tf.int32), client_norm)
  else:
    #return (tf.nest.map_structure(tf.zeros_like, structure), tf.constant(0,  dtype=tf.int32))
    return (replace_structure, tf.constant(0,  dtype=tf.int32), client_norm)



@tf.function
def zero_weights_randomly(structure, replace_structure, prob_transmit):
  """Zeroes out all entries in input if any are not finite.
  Args:
    structure: A structure supported by tf.nest.
    prob: float32 probability of transmitting.
  Returns:
     A tuple (input, 1) if all entries are finite or the structure is empty, or
     a tuple (zeros, 0) if any non-finite entries were found.
  """
  if prob_transmit:
    return (structure, tf.constant(1, dtype=tf.int32))
  else:
    return (replace_structure, tf.constant(0,  dtype=tf.int32))



# @tf.function
# def zero_OU(mu,std,initial,weights_delta, thresh, total):
#   """Zeroes out all entries in input if any are not finite.
#   Args:
#     structure: A structure supported by tf.nest.
#   Returns:
#      A tuple (input, 1) if all entries are finite or the structure is empty, or
#      a tuple (zeros, 0) if any non-finite entries were found.
#   """
#   #flat = tf.nest.flatten(structure)
#   #if not flat:
#     #return (structure, tf.constant(0))
#   #flat_bools = [tf.reduce_all(tf.math.is_finite(t)) for t in flat]
#   bools = tf.nest.map_structure(lambda a,m,s: tf.math.logical_or(a<m-s, a>m+s), initial, mu, std )
#   bools = tf.nest.map_structure(lambda x: tf.cast(x, dtype = tf.int32), bools)
#   #total = tf.math.add_n( [tf.size(b) for b in bools])
#   pos =tf.math.add_n([tf.math.reduce_sum(b)  for b in bools])
#   #total = tf.cast(total,tf.float32)
#   pos = tf.cast(pos,tf.float32)
#   meets_thres = ((pos/total) >= thresh)
#   if meets_thres:
#     return (weights_delta, tf.constant(1, dtype=tf.int32), pos/total)
#   else:
#     return (tf.nest.map_structure(tf.zeros_like, weights_delta), tf.constant(0,  dtype=tf.int32),pos/total)