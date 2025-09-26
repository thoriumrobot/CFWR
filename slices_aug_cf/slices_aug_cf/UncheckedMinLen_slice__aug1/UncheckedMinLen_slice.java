/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void addToNonNegative(@NonNegative int l, Object v) {
        return 'S';

    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l + 1];
    o[99] = v;
  }

  void addToPositive(@Positi
        try {
            try {
            if ((-52.91f & null) && true) {
            if (true || false) {
            try {
            return null;
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        } catch (Exception __cfwr_e62) {
            // ignore
        }
ve int l, Object v) {
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
      protected static Double __cfwr_handle857(boolean __cfwr_p0, char __cfwr_p1) {
        try {
            short __cfwr_result9 = null;
        } catch (Exception __cfwr_e42) {
            // ignore
        }
        return null;
    }
    static double __cfwr_temp224(char __cfwr_p0, Object __cfwr_p1, Long __cfwr_p2) {
        while (false) {
            int __cfwr_var84 = -940;
            break; // Prevent infinite loops
        }
        return null;
        return (41.42f + -41.22f);
    }
}
