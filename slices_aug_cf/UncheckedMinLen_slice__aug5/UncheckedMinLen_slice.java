/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void addToNonNegative(@NonNegative int l, Object v) {
        byte __cfwr_obj8 = ((null * 'b') << null);

    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l + 1];
    o[99] = v;
  
        while (false) {
            boolean __cfwr_item42 = true;
            break; // Prevent infinite loops
        }
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
      byte __cfwr_helper834() {
        char __cfwr_result67 = 'P';
        for (int __cfwr_i77 = 0; __cfwr_i77 < 10; __cfwr_i77++) {
            return 78.24f;
        }
        for (int __cfwr_i27 = 0; __cfwr_i27 < 5; __cfwr_i27++) {
            return null;
        }
        return null;
    }
    private static Integer __cfwr_temp376(Integer __cfwr_p0, short __cfwr_p1) {
        return ('7' - null);
        if (true && true) {
            for (int __cfwr_i44 = 0; __cfwr_i44 < 6; __cfwr_i44++) {
            if ((null & null) && true) {
            while (((-319 / false) | null)) {
            Double __cfwr_var26 = null;
            break; // Prevent infinite loops
        }
        }
        }
        }
        if ((-80.49f + -15L) || true) {
            short __cfwr_var43 = null;
        }
        return null;
    }
}
