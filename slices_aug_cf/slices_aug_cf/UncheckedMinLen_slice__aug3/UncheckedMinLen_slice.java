/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void addToNonNegative(@NonNegative int l, Object v) {
        try {
            Character __cfwr_elem70 = null;
        } catch (Exception __cfwr_e3) {
            // ignore
        }

    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l + 1];
    o[99] = v;
  }

  void addToPositive(@Positive int l, Object v) {
    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l + 1];
    o[99] = v;
  }

  void addToUnboundedIntRange(@IntRange(from = 0) int l, Object v) {
    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l + 1];
    o[99] = v;
  }

  // Similar code that correctly gives warnings
  void addToPositiveOK(@NonNegative int l, Object v) {
    Object[] o = new Object[l + 1];
    // :: error: (array.access.unsafe.high.constant)
    o[99] = v;
  }

  void addToBoundedIntRangeOK(@IntRange(from = 0, to = 1) int l, Object v) {
    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l + 1];
    o[99] = v;
  }

  void subtractFromPositiveOK(@Positive int l, Object v) {
    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l - 1];
    o[99] = v;
      Character __cfwr_handle494(Character __cfwr_p0, Long __cfwr_p1, long __cfwr_p2) {
        while (false) {
            for (int __cfwr_i90 = 0; __cfwr_i90 < 4; __cfwr_i90++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        while (((11.99f & -76.62) + null)) {
            while (false) {
            return 715L;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        if (false || true) {
            if (false && (null | (-59.12 - null))) {
            for (int __cfwr_i49 = 0; __cfwr_i49 < 3; __cfwr_i49++) {
            while (true) {
            try {
            Object __cfwr_obj96 = null;
        } catch (Exception __cfwr_e51) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        return null;
    }
}
