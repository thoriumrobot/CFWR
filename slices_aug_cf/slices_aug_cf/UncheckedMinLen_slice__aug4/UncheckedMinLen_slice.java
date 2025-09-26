/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void addToNonNegative(@NonNegative int l, Object v) {
        while ((false * 256)) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 5; __cfwr_i61++) {
            if (((36.98f | null) + 'j') || true) {
            Object __cfwr_item96 = null;
        }
        }
            break; // Prevent infinite loops
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
      private Long __cfwr_util514() {
        try {
            while (false) {
            while ((99.31 / (-3.03 >> -92.11))) {
            short __cfwr_entry83 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        try {
            if (false && true) {
            try {
            Character __cfwr_obj18 = null;
        } catch (Exception __cfwr_e77) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        try {
            Boolean __cfwr_data33 = null;
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        return null;
        return null;
    }
}
