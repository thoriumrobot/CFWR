/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void addToNonNegative(@NonNegative int l, Object v) {
        for (int __cfwr_i74 = 0; __cfwr_i74 < 2; __cfwr_i74++) {
            while (false) {
            boolean __cfwr_node45 = true;
            break; // Prevent infinite loops
        }
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
      Double __cfwr_func942() {
        try {
            while (false) {
            try {
            if (false || ((67.25f & 'L') & (87.30 | 'j'))) {
            return 52.67;
        }
        } catch (Exception __cfwr_e27) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e6) {
            // ignore
        }
        try {
            return null;
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        for (int __cfwr_i12 = 0; __cfwr_i12 < 8; __cfwr_i12++) {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 3; __cfwr_i97++) {
            while (true) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 10; __cfwr_i14++) {
            try {
            return null;
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        return null;
    }
    public static Integer __cfwr_process797() {
        while (false) {
            while ((false % (860 - 523L))) {
            long __cfwr_val61 = -146L;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        if (((null | null) / -531L) && true) {
            String __cfwr_item72 = "data74";
        }
        while (false) {
            if ((('8' * null) | -20.49) && true) {
            return null;
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
