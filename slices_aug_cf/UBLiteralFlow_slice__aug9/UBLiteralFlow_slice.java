/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private static @IndexOrLow("#1") int lineStartIndexPartial(
      String s, @GTENegativeOne int lineStart) {
        try {
            for (int __cfwr_i82 = 0; __cfwr_i82 < 10; __cfwr_i82++) {
            while (false) {
            Object __cfwr_item37 = null;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e65) {
            // ignore
        }

    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    return result;
  }

  private static @LTLengthOf("#1") int lineStartIndexPartial2(
      String s, @GTENegativeOne int lineStart) {
    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    return result;
  }

  private static @LTLengthOf(value = "#1", offset = "1") int lineStartIndexPartial3(
      String s, @GTENegativeOne int lineStart) {
    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    // :: error: (return)
    return result;
  }

  private static @LTLengthOf(value = "#1", offset = "-1") int lineStartIndexPartial4(
      String s, @GTENegativeOne int lineStart) {
    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    return result;
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @IndexOrLow("#1") int lineStartIndex(String s, int start) {
    if (s.length() == 0) {
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      return lineStart;
    }
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @LTLengthOf("#1") int lineStartIndex2(String s, int start) {
    if (s.length() == 0) {
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      return lineStart;
    }
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @LTLengthOf(value = "#1", offset = "1") int lineStartIndex3(String s, int start) {
    if (s.length() == 0) {
      // :: error: (return)
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      // :: error: (return)
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      // :: error: (return)
      return lineStart;
    }
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @LTLengthOf(value = "#1", offset = "-1") int lineStartIndex4(String s, int start) {
    if (s.length() == 0) {
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      return lineStart;
    }
      protected static long __cfwr_func712(Integer __cfwr_p0) {
        for (int __cfwr_i31 = 0; __cfwr_i31 < 2; __cfwr_i31++) {
            for (int __cfwr_i81 = 0; __cfwr_i81 < 1; __cfwr_i81++) {
            try {
            return null;
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        }
        }
        while (false) {
            return 63.33;
            break; // Prevent infinite loops
        }
        if ((-613 + 805) || true) {
            for (int __cfwr_i84 = 0; __cfwr_i84 < 1; __cfwr_i84++) {
            while ((null ^ (null << null))) {
            while (false) {
            while (true) {
            double __cfwr_node67 = -58.95;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        return -513L;
    }
}
