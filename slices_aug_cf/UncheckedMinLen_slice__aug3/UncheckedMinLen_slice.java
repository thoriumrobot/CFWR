/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void addToNonNegative(@NonNegative int l, Object v) {
        if (true && (-106L ^ null)) {
            Integer __cfwr_node55 = null;
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
      protected static short __cfwr_func895(short __cfwr_p0, Boolean __cfwr_p1, short __cfwr_p2) {
        for (int __cfwr_i99 = 0; __cfwr_i99 < 8; __cfwr_i99++) {
            if (true || ((3.37 & -668L) ^ 'a')) {
            return 629;
        }
        }
        while (false) {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 2; __cfwr_i12++) {
            while ((92.04f | 84.75)) {
            if (true || true) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i60 = 0; __cfwr_i60 < 1; __cfwr_i60++) {
            if (false || true) {
            while (false) {
            Long __cfwr_result98 = null;
            break; // Prevent infinite loops
        }
        }
        }
        return (88.64f >> null);
    }
}
