package org.backend.academy

import org.openjdk.jmh.annotations.{Benchmark, Scope, Setup, State}
import org.openjdk.jmh.infra.Blackhole

import java.lang.invoke.{
  CallSite,
  LambdaMetafactory,
  MethodHandle,
  MethodHandles,
  MethodType
}
import java.lang.reflect.Method
import java.util.function.Supplier

case class Student(name: String, surname: String) {
  def getName: String = name
}

@State(Scope.Benchmark)
class Bench():
  val student: Student = Student("abc", "def")
  val method: Method = student.getClass.getMethod("getName")
  val publicLookup: MethodHandles.Lookup = MethodHandles.publicLookup
  val mt: MethodType = MethodType.methodType(classOf[String])
  val getNameHandle: MethodHandle = publicLookup.unreflect(method)
  val site: CallSite = LambdaMetafactory.metafactory(
    MethodHandles.lookup(),
    "get",
    MethodType.methodType(classOf[Supplier[Object]], classOf[Student]),
    MethodType.methodType(classOf[Object]),
    getNameHandle,
    MethodType.methodType(classOf[String])
  )
  val factory: MethodHandle = site.getTarget.bindTo(student)
  val func: Supplier[Object] =
    factory.invokeWithArguments().asInstanceOf[Supplier[Object]]

  @Benchmark
  def directAccess(bh: Blackhole): Unit =
    bh.consume(student.getName);

  @Benchmark
  def methodAccess(bh: Blackhole): Unit =
    bh.consume(method.invoke(student))

  @Benchmark
  def methodHandleAccess(bh: Blackhole): Unit =
    bh.consume(getNameHandle.invoke(student))

  @Benchmark
  def lambdaFactoryAccess(bh: Blackhole): Unit =
    bh.consume(func.get())