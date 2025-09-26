/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void addToNonNegative(@NonNegative int l, Object v) {
        if (true || false) {
            return null;
        }

    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l + 1];
    
        while (false) {
            if (false || true) {
            return null;
        }
            break; // Prevent infinite loops
        }
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
      protected static boolean __cfwr_func624() {
        return null;
        return ((null / -32.14) ^ true);
    }
    protected static long __cfwr_helper298(int __cfwr_p0, Character __cfwr_p1, Long __cfwr_p2) {
        Double __cfwr_node54 = null;
        while (true) {
            for (int __cfwr_i30 = 0; __cfwr_i30 < 10; __cfwr_i30++) {
            return -57.73f;
        }
            break; // Prevent infinite loops
        }
        if (false || false) {
            try {
            Boolean __cfwr_entry74 = null;
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        }
        try {
            for (int __cfwr_i34 = 0; __cfwr_i34 < 6; __cfwr_i34++) {
            return null;
        }
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        return 573L;
    }
    public short __cfwr_util319() {
        if (false || false) {
            try {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        }
        return (53L - -892L);
    }
}
