/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void addToNonNegative(@NonNegative int l, Object v) {
        Float __cfwr_var44 = null;

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
      public Boolean __cfwr_temp727(double __cfwr_p0) {
        for (int __cfwr_i40 = 0; __cfwr_i40 < 4; __cfwr_i40++) {
            try {
            while (false) {
            for (int __cfwr_i86 = 0; __cfwr_i86 < 2; __cfwr_i86++) {
            while (true) {
            for (int __cfwr_i40 = 0; __cfwr_i40 < 8; __cfwr_i40++) {
            return -70.75f;
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        }
        try {
            if ((413L >> (false * null)) && true) {
            while (false) {
            Integer __cfwr_data58 = null;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e4) {
            // ignore
        }
        Float __cfwr_entry97 = null;
        return null;
    }
    protected int __cfwr_process772(Object __cfwr_p0, double __cfwr_p1) {
        Float __cfwr_elem22 = null;
        return 606;
    }
}
